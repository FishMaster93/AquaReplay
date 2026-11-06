"""
压缩视频修复网络：把 H.264 解码后的片段修复得更接近原始片段，只用在记忆回放上。

逐帧 2D 残差 CNN，输入当前帧和前后各一帧（共 9 通道），输出当前帧的修正量。
两种用法：
  共享：只在 Phase 0 的训练片段上训练一次（scripts/train_restorer.py），之后冻结，所有物种共用。
        实验发现它对 Phase 0 物种的回放修复得好，对后来外观不同的物种帮助消失。
  每物种：学完物种 k 后，用 k 的训练片段（压缩版 -> 原始版）从上一个修复网络初始化再训练一个，训练完冻结，
        只用来修复物种 k 的记忆样本（fit_restorer）。每个网络约 13 万参数（fp16 约 0.26 MB），计入记忆预算。
损失：L1 + w·(1 − cos(h(R(x̃)), h(x)))，h 是当时冻结的视频模型（dropout 之前的特征），即任务感知修复。
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Restorer(nn.Module):
    def __init__(self, ch=48, depth=8):
        super().__init__()
        layers = [nn.Conv2d(9, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(ch, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """x: (B,T,H,W,3) float in [0,1] -> 同形状"""
        B, T, H, W, _ = x.shape
        f = x.permute(0, 1, 4, 2, 3)                                       # (B,T,3,H,W)
        prev = torch.cat([f[:, :1], f[:, :-1]], 1)
        nxt = torch.cat([f[:, 1:], f[:, -1:]], 1)
        inp = torch.cat([prev, f, nxt], 2).reshape(B * T, 9, H, W)
        out = f.reshape(B * T, 3, H, W) + self.net(inp)
        return out.reshape(B, T, 3, H, W).permute(0, 1, 3, 4, 2)

    @torch.no_grad()
    def restore_uint8(self, x):
        """x: (B,T,H,W,3) uint8 cuda -> uint8"""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = self(x.float() / 255)
        return (y.float() * 255).round().clamp(0, 255).to(torch.uint8)


def fit_restorer(net, model, loader, steps, w=0.2, lr=1e-4, log=print):
    """net: 初始化好的 Restorer（会被原地训练）；model: 当前视频模型（复制一份冻结使用）；
    loader 产出 (原始片段, 压缩版, 标签)，uint8。返回冻结的 net。"""
    h = copy.deepcopy(model).eval()
    for q in h.parameters():
        q.requires_grad_(False)
    net.train()
    for q in net.parameters():
        q.requires_grad_(True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    step = 0
    while step < steps:
        for x, xc, _ in loader:
            x, xc = x.cuda(non_blocking=True).float() / 255, xc.cuda(non_blocking=True).float() / 255
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = net(xc)
                with torch.no_grad():
                    hx = h(x.permute(0, 1, 4, 2, 3))[1].float()
                hy = h(y.clamp(0, 1).permute(0, 1, 4, 2, 3))[1].float()
            l1 = F.l1_loss(y.float(), x)
            feat = (1 - F.cosine_similarity(hy, hx, dim=1)).mean()
            loss = l1 + w * feat
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0 or step == steps:
                log(f'    restorer step {step}/{steps} l1 {l1.item() * 255:.3f} feat {feat.item():.4f}')
            if step >= steps:
                break
    del h
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net


def restorer_nbytes(net):
    return 2 * sum(q.numel() for q in net.parameters())


def load_restorer(path):
    m = Restorer().cuda().eval()
    m.load_state_dict(torch.load(path, map_location='cuda'))
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float(), reduction='none').flatten(1).mean(1)
    return (10 * torch.log10(255 ** 2 / mse.clamp_min(1e-6))).mean()
