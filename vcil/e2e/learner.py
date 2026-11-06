"""
端到端（编码器不冻结）的有样本物种增量学习。所有方法共用同一个模型（S3D + CORAL 强度头，从 Phase 0
checkpoint 开始）、同一个训练循环和同一个按字节计的记忆预算，只在下面几个开关上不同：

  codec       : 记忆样本的存储格式（见 codecs.py），None 表示不存样本
  consistency : 压缩一致性训练——当前片段 x 与其压缩版 x̃ = decode(encode(x)) 一起训练，
                L += [L_cls(x̃)] + λ·D(x̃, sg[x])（vCLIMB 的时间一致性即 codec=少帧时的特例）
  cons_type   : D 的形式。logit = ‖z(x̃) − z(x)‖²（v1，作用在 CORAL logits 上，受 dropout 噪声和 logit
                尺度影响）；prob = 累积概率 σ(z) 上的 BCE（有界）；feat = dropout 之前的特征上 1 − cos
  cons_cls    : 是否对 x̃ 也加分类损失 L_cls(x̃)
  select      : 记忆选样 random / kcenter / boundary（强度感知的不确定性边界样本 + 代表样本 + 去重）
  der         : DER++ 的 logits 蒸馏 L += α·‖f(x_m) − z_m‖²
  anchor      : 原始特征锚点——每个记忆样本另存插入时原始片段的特征 a_m（fp16，计入预算），
                回放时 L += μ·(1 − cos(h(x̂_m), a_m))，x̂_m 是解码出的片段，h 是 dropout 之前的特征
  restore     : 修复网络权重路径（vcil/e2e/restore.py）；回放的解码片段先修复再送进模型
  freeze_bn   : 训练时 BN 层保持 eval（用 Phase 0 的统计量）
  refit_epochs: >0 时每个阶段训练完后，用当前骨干提取当前物种片段 + 全部记忆样本（解码、修复后）的特征，
                在特征上把分类头再训练 refit_epochs 轮（Adam lr 1e-3，batch 256）。按物种均衡采样
                （每个物种被抽到的总概率相同），否则当前物种的上万个片段会压倒记忆里少量的旧物种样本。冻结特征上 MLP 头要训练
                约 30 轮才收敛（3 轮 75.4%，10 轮 82.2%，30 轮 86.2%），端到端每阶段 3 轮远远不够；
                存视频而不是特征的好处也在这里：骨干更新后记忆样本的特征可以重新提取
  refit_balance: False 时重拟合按样本均匀采样（消融用）
  tune_from   : 只微调 S3D 第 tune_from 个模块及之后（如 14 = Mixed_5b/5c），之前的模块冻结且保持 eval
  restore_per_species : 每学完一个物种就训练（并冻结）该物种自己的修复网络，从上一个初始化；
                restore 给出的权重用作物种 0 的修复网络。修复网络权重计入记忆预算
  align       : >0 时加强度对齐的跨物种迁移损失（权重）。每个强度一个原型 = 记忆（旧物种）样本特征的滑动平均
                （不存储、不回传梯度）；当前物种的特征对原型做 softmax(cos/τ)，目标是序数软标签
                q_c ∝ exp(−|c − y| / σ)，让新物种落到旧物种已学好的强度结构上，且相邻强度比远处强度更近
  align_tau, align_sigma, align_m : τ（0.1）、σ（0.5）、原型动量（0.9）
  kd_feat, kd_out : 师生蒸馏（权重）。每个阶段开始时复制当前模型作冻结教师，当前片段和记忆片段一起送进师生，
                L += kd_feat·(1 − cos(h_s, h_t)) + kd_out·BCE(z_s, σ(z_t))，h 为特征、z 为 CORAL logits
  ewc         : >0 时加 EWC 正则（权重 λ）。每个阶段结束后在当前物种上估计对角 Fisher（逐 batch 梯度平方，
                最多 ewc_batches 个 batch），与之前的累加；训练时 L += λ/2·Σ F(θ − θ*)²，θ* 为上一阶段结束时的参数
  nme         : iCaRL 的最近类均值分类：每个阶段结束后用记忆样本（解码、修复后）的归一化特征算每个强度的均值，
                预测时取余弦最近的强度，代替分类头
  sp_thresh   : 物种相关的序数阈值。头输出的 3 个 CORAL logits 加上每个物种自己的偏移 b_s（3 个数/物种，
                零初始化，只在重拟合时和头一起学），即共享的"激烈程度"、每个物种各自的强度分界。
                1 = 测试时用真实物种；2 = 测试时用线性物种分类器（冻结特征上、与头一起重拟合）预测物种

每个阶段：用当前物种训练（每个 batch = 当前片段 bs 个 [+ 其压缩版] + 记忆重建片段 bs 个）→ 选样、编码、入库。
Stage 0 不训练（Phase 0 checkpoint 已在其训练集上适应），只建记忆。
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from vcil.data import VideoClipDataset, gpu_augment
from vcil.e2e.codecs import build_codec
from vcil.e2e.memory import VideoMemory
from vcil.e2e.restore import Restorer, fit_restorer, load_restorer, restorer_nbytes
from vcil.models.video import VideoModel, coral_loss, coral_predict

NUM_LEVELS = 4
MAX_SPECIES = 8

# 各方法的开关；codec 参数在同一预算下取各自论文/常用设置
METHODS = {
    'finetune': dict(codec=None),
    'er':       dict(codec='raw'),
    'der':      dict(codec='raw', der=True),
    'smile':    dict(codec='frame'),                          # SMILE：每段只存 1 帧
    'vclimb':   dict(codec='subsamp4', consistency=True),     # vCLIMB：少帧存储 + 时间一致性
    'mrdc':     dict(codec='jpeg50'),                         # MRDC：逐帧 JPEG
    'er_h264':  dict(codec='h264crf33'),                      # 视频编码基线
    'icarl':    dict(codec='raw', select='herding', kd_out=1.0, nme=True),   # iCaRL：herding 选样 + 蒸馏 + 最近类均值
    'aquareplay': dict(codec='h264crf33', restore_per_species=True, refit_epochs=30, refit_balance=False),  # 本文方法
}


class _PairDataset(Dataset):
    """返回原始片段及其压缩版（在 worker 里编解码）。"""

    def __init__(self, items, codec):
        self.items, self.codec = items, codec

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        clip = torch.from_numpy(np.load(it['video']))
        return clip, self.codec.decode(self.codec.encode(clip)[0]), it['label']


class E2ELearner:
    def __init__(self, method, ckpt, budget_mb=200, epochs=3, lr=1e-4, bs=12, lam=1.0, der_alpha=0.5,
                 seed=42, num_workers=8, mu=1.0, **override):
        cfg = {'codec': None, 'consistency': False, 'cons_type': 'logit', 'cons_cls': True, 'select': 'random', 'der': False,
               'anchor': False, 'restore': None, 'restore_per_species': False, 'restore_steps': 1500,
               'restore_w': 0.2, 'freeze_bn': False, 'tune_from': None, 'refit_epochs': 0, 'refit_balance': True,
               'align': 0.0, 'align_tau': 0.1, 'align_sigma': 0.5, 'align_m': 0.9, 'sp_thresh': 0, 'kd_feat': 0.0, 'kd_out': 0.0, 'ewc': 0.0, 'ewc_batches': 100, 'nme': False, 'alloc': 'cell', **METHODS[method], **override}
        self.cfg, self.epochs, self.lr, self.bs, self.lam, self.der_alpha = cfg, epochs, lr, bs, lam, der_alpha
        self.mu = mu
        self.num_workers = num_workers
        torch.manual_seed(seed)
        np.random.seed(seed)
        state = torch.load(ckpt, map_location='cuda', weights_only=False)
        self.model = VideoModel(mlp=state.get('mlp', False), backbone=state.get('backbone', 's3d')).cuda()
        self.model.load_state_dict(state['model_state_dict'])
        self.frozen = list(self.model.backbone.base)[:cfg['tune_from']] if cfg['tune_from'] else []
        for mod in self.frozen:
            for q in mod.parameters():
                q.requires_grad_(False)
        self.codec = build_codec(cfg['codec']) if cfg['codec'] else None
        self.memory = (VideoMemory(self.codec, budget_mb, cfg['select'], seed, num_workers, cfg['anchor'], cfg['alloc'])
                       if self.codec else None)
        self.restorer = load_restorer(cfg['restore']) if cfg['restore'] else None
        self.restorers = {}        # 每物种修复网络：k -> Restorer
        self.sp_bias = torch.zeros(MAX_SPECIES, NUM_LEVELS - 1, device='cuda', requires_grad=True)
        self.sp_clf = torch.nn.Linear(self.model.backbone.dim, MAX_SPECIES).cuda()
        self.seen = 0
        self.fisher, self.anchor_params = None, None

    def _forward(self, x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, emb = self.model(x)
        return logits.float(), emb.float()

    def learn(self, k, items, log=print):
        self.seen = k + 1
        if k > 0 and self.epochs:          # epochs=0：编码器全程冻结，只建记忆、训练修复网络和重拟合头
            self._train(items, log)
        if self.cfg['restore_per_species']:
            self._fit_species_restorer(k, items, log)
        # boundary 选样要用重拟合后的头算 CORAL 概率，所以先重拟合再入库；其他选样保持先入库再重拟合
        refit_first = self.cfg['select'] == 'boundary'
        if self.cfg['refit_epochs'] and k > 0 and refit_first:
            self.refit_head(items, k, log)
        if self.memory is not None:
            emb, logits = self.embed(items)
            self.memory.add_species(k, items, emb, logits)
            log(f'  memory: {self.memory.stats()}')
        if self.cfg['refit_epochs'] and k > 0 and not refit_first:
            self.refit_head(items, k, log)
        if self.cfg['ewc']:
            self._update_fisher(items, log)
        if self.cfg['nme'] and self.memory is not None and len(self.memory):
            self._update_class_means(log)

    @torch.no_grad()
    def _update_class_means(self, log):
        self.model.eval()
        es, ys = [], []
        for mx, my, _, _, mk in self.memory.sequential_loader(16):
            es.append(F.normalize(self._forward(self._restore(mx.cuda(non_blocking=True), mk))[1], dim=1).cpu())
            ys.append(my)
        e, y = torch.cat(es), torch.cat(ys)
        self.class_means = torch.stack([F.normalize(e[y == c].mean(0), dim=0) if (y == c).any()
                                        else torch.zeros(e.shape[1]) for c in range(NUM_LEVELS)]).cuda()
        log(f'  nme: class means from {len(e)} exemplars')

    def _update_fisher(self, items, log):
        params = {n: q for n, q in self.model.named_parameters() if q.requires_grad}
        fisher = {n: torch.zeros_like(q) for n, q in params.items()}
        loader = DataLoader(VideoClipDataset(items), batch_size=self.bs, shuffle=True, num_workers=self.num_workers,
                            pin_memory=True, drop_last=True)
        self.model.eval()
        nb = 0
        for x, y, _ in loader:
            logits, _ = self._forward(x.cuda(non_blocking=True))
            self.model.zero_grad(set_to_none=True)
            coral_loss(logits, y.cuda(non_blocking=True)).backward()
            for n, q in params.items():
                if q.grad is not None:
                    fisher[n] += q.grad.detach() ** 2
            nb += 1
            if nb >= self.cfg['ewc_batches']:
                break
        self.model.zero_grad(set_to_none=True)
        fisher = {n: f / nb for n, f in fisher.items()}
        self.fisher = fisher if self.fisher is None else {n: self.fisher[n] + fisher[n] for n in fisher}
        self.anchor_params = {n: q.detach().clone() for n, q in params.items()}
        log(f'  ewc: fisher from {nb} batches, mean {torch.cat([f.flatten() for f in self.fisher.values()]).mean().item():.3e}')

    def refit_head(self, items, k=None, log=print):
        xs, ys = [self.embed(items)[0]], [torch.tensor([it['label'] for it in items])]
        ks = [torch.full((len(items),), -1 if k is None else k)]
        if self.memory is not None and len(self.memory):
            self.model.eval()
            with torch.no_grad():
                for mx, my, _, _, mk in self.memory.sequential_loader(16):
                    xs.append(self._forward(self._restore(mx.cuda(non_blocking=True), mk))[1].cpu())
                    ys.append(my)
                    ks.append(mk)
        x, y, sp = torch.cat(xs).cuda(), torch.cat(ys).cuda(), torch.cat(ks).cuda()
        _, inv, cnt = sp.unique(return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv].float() if self.cfg['refit_balance'] else torch.ones_like(sp, dtype=torch.float)   # 每个物种总权重相同
        head = self.model.head
        spt = self.cfg['sp_thresh'] and k is not None
        params = list(head.parameters()) + ([self.sp_bias] + list(self.sp_clf.parameters()) if spt else [])
        opt = torch.optim.Adam(params, lr=1e-3)
        g = torch.Generator(device='cuda').manual_seed(0)
        for _ in range(self.cfg['refit_epochs']):
            head.train()
            perm = torch.multinomial(w, len(x), replacement=True, generator=g)
            for i in range(0, len(x), 256):
                j = perm[i:i + 256]
                if spt:
                    loss = coral_loss(head(x[j]) + self.sp_bias[sp[j]], y[j]) + F.cross_entropy(self.sp_clf(x[j]), sp[j])
                else:
                    loss = coral_loss(head(x[j]), y[j])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        head.eval()
        log(f'  refit head on {len(x)} features ({self.cfg["refit_epochs"]} epochs), last loss {loss.item():.4f}')

    def _fit_species_restorer(self, k, items, log):
        if k == 0 and self.restorer is not None:
            net = self.restorer
        else:
            net = Restorer().cuda()
            prev = self.restorers.get(k - 1, self.restorer)
            if prev is not None:
                net.load_state_dict(prev.state_dict())
            loader = DataLoader(_PairDataset(items, self.codec), batch_size=2, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True, drop_last=True)
            net = fit_restorer(net, self.model, loader, self.cfg['restore_steps'], self.cfg['restore_w'], log=log)
        self.restorers[k] = net
        self.memory.reserved += restorer_nbytes(net)

    def _restore(self, mx, mk):
        if self.cfg['restore_per_species']:
            mk = mk.cuda(non_blocking=True)
            for k in mk.unique().tolist():
                sel = mk == k
                mx[sel] = self.restorers[k].restore_uint8(mx[sel])
            return mx
        return self.restorer.restore_uint8(mx) if self.restorer is not None else mx

    def _train(self, items, log):
        cons = self.cfg['consistency']
        ds = _PairDataset(items, self.codec) if cons else VideoClipDataset(items)
        loader = DataLoader(ds, batch_size=self.bs, shuffle=True, num_workers=self.num_workers, pin_memory=True,
                            drop_last=True, persistent_workers=True, prefetch_factor=4)
        kd = self.cfg['kd_feat'] or self.cfg['kd_out']
        if kd:
            teacher = copy.deepcopy(self.model).eval()
            for q in teacher.parameters():
                q.requires_grad_(False)
        mem_iter = iter(self.memory.loader(self.bs)) if self.memory is not None and len(self.memory) else None
        protos = None                      # 强度原型 (C, D)，每个阶段从记忆样本重新累积
        opt = torch.optim.Adam([q for q in self.model.parameters() if q.requires_grad], lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs * len(loader))
        for ep in range(self.epochs):
            self.model.train()
            for mod in self.frozen:
                mod.eval()
            if self.cfg['freeze_bn']:          # BN 用 Phase 0 的统计量，不随小 batch 更新
                for mod in self.model.modules():
                    if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                        mod.eval()
            tot, n = 0.0, 0
            for step, batch in enumerate(loader):
                if cons:
                    x, xc, y = batch
                    x, xc = gpu_augment(x.cuda(non_blocking=True), xc.cuda(non_blocking=True))
                    parts = [x, xc]
                else:
                    x, y = batch[0], batch[1]
                    parts = [gpu_augment(x.cuda(non_blocking=True))]
                y = y.cuda(non_blocking=True)
                if mem_iter is not None:
                    mx, my, mz, ma, mk = next(mem_iter)
                    mx = self._restore(mx.cuda(non_blocking=True), mk)
                    parts.append(gpu_augment(mx))
                xin = torch.cat(parts)
                logits, emb = self._forward(xin)
                lx = logits[:self.bs]
                loss = coral_loss(lx, y)
                if kd:
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        tl, te = teacher(xin)
                    if self.cfg['kd_feat']:
                        loss = loss + self.cfg['kd_feat'] * (1 - F.cosine_similarity(emb, te.float(), dim=1)).mean()
                    if self.cfg['kd_out']:
                        loss = loss + self.cfg['kd_out'] * F.binary_cross_entropy_with_logits(logits, torch.sigmoid(tl.float()))
                if cons:
                    lc = logits[self.bs:2 * self.bs]
                    if self.cfg['cons_cls']:
                        loss = loss + coral_loss(lc, y)
                    loss = loss + self.lam * self._cons(lx, lc, emb[:self.bs], emb[self.bs:2 * self.bs])
                if mem_iter is not None:
                    lm = logits[-self.bs:]
                    my, mz = my.cuda(non_blocking=True), mz.cuda(non_blocking=True)
                    loss = loss + coral_loss(lm, my)
                    if self.cfg['der']:
                        loss = loss + self.der_alpha * F.mse_loss(lm, mz)
                    if self.cfg['anchor']:
                        em = emb[-self.bs:]
                        loss = loss + self.mu * (1 - F.cosine_similarity(em, ma.cuda(non_blocking=True), dim=1)).mean()
                    if self.cfg['align']:
                        protos = self._update_protos(protos, emb[-self.bs:].detach(), my)
                        loss = loss + self.cfg['align'] * self._align(emb[:self.bs], y, protos)
                if self.cfg['ewc'] and self.fisher is not None:
                    pen = sum((self.fisher[n] * (q - self.anchor_params[n]) ** 2).sum()
                              for n, q in self.model.named_parameters() if n in self.fisher)
                    loss = loss + self.cfg['ewc'] / 2 * pen
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot, n = tot + loss.item(), n + 1
                if step % 200 == 0:
                    log(f'    epoch {ep} step {step}/{len(loader)} loss {loss.item():.4f}')
            log(f'  epoch {ep}: mean loss {tot / n:.4f}')

    def _update_protos(self, protos, em, my):
        C, m = NUM_LEVELS, self.cfg['align_m']
        if protos is None:
            protos = torch.full((C, em.shape[1]), float('nan'), device=em.device)
        em = F.normalize(em, dim=1)
        for c in my.unique().tolist():
            mean = em[my == c].mean(0)
            protos[c] = mean if torch.isnan(protos[c, 0]) else m * protos[c] + (1 - m) * mean
        return protos

    def _align(self, e, y, protos):
        ok = ~torch.isnan(protos[:, 0])
        if ok.sum() < 2:
            return e.new_zeros(())
        levels = torch.arange(NUM_LEVELS, device=e.device)[ok]
        sim = F.normalize(e, dim=1) @ F.normalize(protos[ok], dim=1).T / self.cfg['align_tau']
        q = torch.softmax(-(levels[None] - y[:, None]).abs().float() / self.cfg['align_sigma'], dim=1)
        return -(q * F.log_softmax(sim, dim=1)).sum(1).mean()

    def _cons(self, lx, lc, ex, ec):
        t = self.cfg['cons_type']
        if t == 'logit':
            return F.mse_loss(lc, lx.detach())
        if t == 'prob':
            return F.binary_cross_entropy_with_logits(lc, torch.sigmoid(lx.detach()))
        return (1 - F.cosine_similarity(ec, ex.detach(), dim=1)).mean()

    @torch.no_grad()
    def embed(self, items, bs=64):
        """原始片段上的特征和 logits（CPU），供选样和 DER++ 使用。"""
        self.model.eval()
        embs, logits = [], []
        for x, _, _ in DataLoader(VideoClipDataset(items), batch_size=bs, num_workers=self.num_workers, pin_memory=True):
            l, e = self._forward(x.cuda(non_blocking=True))
            logits.append(l.cpu())
            embs.append(e.cpu())
        return torch.cat(embs), torch.cat(logits)

    @torch.no_grad()
    def predict(self, items, k=None, bs=64):
        """k: 这些片段的物种编号（sp_thresh=1 时使用）"""
        self.model.eval()
        preds = []
        for x, _, _ in DataLoader(VideoClipDataset(items), batch_size=bs, num_workers=self.num_workers, pin_memory=True):
            logits, emb = self._forward(x.cuda(non_blocking=True))
            if self.cfg['sp_thresh'] == 1 and k is not None and k < self.seen:
                logits = logits + self.sp_bias[k]
            elif self.cfg['sp_thresh'] == 2:
                logits = logits + self.sp_bias[self.sp_clf(emb)[:, :self.seen].argmax(1)]
            if self.cfg['nme'] and getattr(self, 'class_means', None) is not None:
                preds.append((F.normalize(emb, dim=1) @ self.class_means.T).argmax(1).cpu())
                continue
            preds.append(coral_predict(logits).cpu())
        return torch.cat(preds)
