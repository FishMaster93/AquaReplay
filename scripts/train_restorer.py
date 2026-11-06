"""
训练压缩视频修复网络（vcil/e2e/restore.py）：Phase 0 训练片段，随机 H.264 档位 crf∈{18,23,28,33} 压缩后解码，
L1 损失修复回原始片段。训练完在 Phase 0 测试集（400 段）上报告各档位的 PSNR 和 Phase 0 模型准确率
（解码 / 修复后 / 原始），结果写 results/pilot/<输出文件名>.json。

任务感知修复（--feat_w > 0）：在完整 224 片段上训练，另加特征损失 w·(1 − cos(h(R(x̃)), h(x)))，
h 是冻结的 Phase 0 S3D（dropout 之前的特征），让修复保留与强度相关的运动细节，而不只是像素接近。
注意此时测试准确率用的是同一个 Phase 0 模型，偏乐观；真正的检验是增量实验。

  python scripts/train_restorer.py --steps 6000
  python scripts/train_restorer.py --feat_w 1 --init workspace/restorer/restorer.pt --steps 4000 --bs 2 \
      --out workspace/restorer/restorer_task1.pt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.data import phase_items
from vcil.e2e.codecs import H264
from vcil.e2e.restore import Restorer, psnr
from vcil.models.video import VideoModel, coral_predict
from vcil.paths import RESULTS, WORKSPACE

CRFS = (18, 23, 28, 33)


class _Pairs(Dataset):
    def __init__(self, items, crfs):
        self.items, self.crfs = items, crfs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip = torch.from_numpy(np.load(self.items[i]['video']))
        crf = self.crfs[int(torch.randint(len(self.crfs), ()))]   # torch 的随机数在每个 worker 里有不同种子
        codec = H264(crf)
        return codec.decode(codec.encode(clip)[0]), clip, self.items[i]['label']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--feat_w', type=float, default=0.0, help='任务感知特征损失权重，0 = 只用 L1')
    p.add_argument('--init', default=None, help='从已有修复网络权重开始')
    p.add_argument('--out', default=str(WORKSPACE / 'restorer' / 'restorer.pt'))
    p.add_argument('--phase0', default=str(WORKSPACE / 'S3D_phase0_clip' / 'phase0_best.pt'),
                   help='Phase 0 适应后的视频模型（特征损失的判别器；也用于测试准确率）')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0)

    st = torch.load(args.phase0, map_location='cuda', weights_only=False)
    model = VideoModel(backbone=st.get('backbone', 's3d')).cuda().eval()
    model.load_state_dict(st['model_state_dict'])
    for q in model.parameters():
        q.requires_grad_(False)

    net = Restorer().cuda()
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location='cuda'))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    loader = DataLoader(_Pairs(phase_items(0, 'train', 'clip'), CRFS), batch_size=args.bs, shuffle=True,
                        num_workers=10, drop_last=True, persistent_workers=True, prefetch_factor=4)
    step, t0 = 0, time.time()
    while step < args.steps:
        for xc, x, _ in loader:
            xc, x = xc.cuda().float() / 255, x.cuda().float() / 255
            if args.feat_w == 0:
                # 只用 L1 时随机裁 112x112 小块训练，省显存、加速
                top, left = np.random.randint(0, 113, 2)
                xc, x = xc[:, :, top:top + 112, left:left + 112], x[:, :, top:top + 112, left:left + 112]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = net(xc)
            l1 = F.l1_loss(y.float(), x)
            loss = l1
            if args.feat_w > 0:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        hx = model(x.permute(0, 1, 4, 2, 3))[1].float()     # float 输入为 (B,T,3,H,W)，[0,1]
                    hy = model(y.clamp(0, 1).permute(0, 1, 4, 2, 3))[1].float()
                feat = (1 - F.cosine_similarity(hy, hx, dim=1)).mean()
                loss = l1 + args.feat_w * feat
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                msg = f' feat {feat.item():.4f}' if args.feat_w > 0 else ''
                print(f'step {step}/{args.steps} l1 {l1.item() * 255:.3f}{msg} | {(time.time() - t0) / 60:.1f} min', flush=True)
            if step >= args.steps:
                break
    torch.save(net.state_dict(), args.out)
    net.eval()
    test = phase_items(0, 'test', 'clip')
    test = [test[i] for i in np.random.RandomState(0).permutation(len(test))[:400]]

    def acc(x, y):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return int((coral_predict(model(x)[0].float()) == y).sum())

    res = {}
    for crf in CRFS:
        n, c_raw, c_dec, c_res, p_dec, p_res = 0, 0, 0, 0, [], []
        for xc, x, y in DataLoader(_Pairs(test, (crf,)), batch_size=16, num_workers=10):
            xc, x, y = xc.cuda(), x.cuda(), y.cuda()
            xr = net.restore_uint8(xc)
            c_raw, c_dec, c_res = c_raw + acc(x, y), c_dec + acc(xc, y), c_res + acc(xr, y)
            p_dec.append(float(psnr(xc, x)))
            p_res.append(float(psnr(xr, x)))
            n += len(y)
        res[crf] = {'acc_raw': c_raw / n, 'acc_decoded': c_dec / n, 'acc_restored': c_res / n,
                    'psnr_decoded': float(np.mean(p_dec)), 'psnr_restored': float(np.mean(p_res))}
        print(f'crf{crf}: ' + ' '.join(f'{k} {v:.4f}' for k, v in res[crf].items()), flush=True)
    out = RESULTS / 'pilot' / (os.path.splitext(os.path.basename(args.out))[0] + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'args': vars(args), 'results': res}, open(out, 'w'), indent=2)


if __name__ == '__main__':
    main()
