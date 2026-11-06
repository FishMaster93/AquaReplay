"""
论文图：修复器效果。
  (a) 一段有代表性的 Strong 测试片段（压缩后 PSNR、PSNR 提升、余弦提升都最接近全体中位数）的同一帧：原始 / H.264 crf33 / 修复后（红框标出局部），
      第二排为红框区域放大（最近邻插值），第三排为 |压缩 − 原始| 与 |修复 − 原始| 误差图（×4，magma，0–1）
  (b) 全部测试片段压缩后 / 修复后的 PSNR 分布
  (c) 全部测试片段 cos(f(x̃), f(x)) 与 cos(f(R(x̃)), f(x)) 的分布
模型与每物种修复器取自 Ours（seed 43）的 checkpoint；f 为学完 7 个物种后的编码器。
输出 paper_figures/fig_restore.{pdf,png}，统计量写 results/pilot/restore_fig_stats.json
  python scripts/make_restore_fig.py [--max_per_species 0]   # 0 = 全部测试片段
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.data import phase_items
from vcil.e2e.codecs import build_codec
from vcil.e2e.learner import _PairDataset
from vcil.e2e.restore import Restorer
from vcil.models.video import VideoModel
from vcil.paths import RESULTS, ROOT, WORKSPACE

CKPT = WORKSPACE / 'e2e_ckpt' / 'aquareplay_b200_main_seed42.pt'
OUT = ROOT / 'paper_figures'
FRAME, BOX, GAIN = 8, 48, 4


def load():
    st = torch.load(CKPT, map_location='cuda', weights_only=False)
    m = VideoModel(mlp=True).cuda().eval()
    m.load_state_dict(st['model_state_dict'])
    rs = {}
    for k, sd in st['restorers'].items():
        r = Restorer().cuda().eval()
        r.load_state_dict(sd)
        rs[int(k)] = r
    return m, rs, st['order']


def psnr(a, b):
    """a, b: (B,T,H,W,3) uint8 -> 每段的 PSNR (B,)"""
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(255 ** 2 / mse.clamp_min(1e-6))


@torch.no_grad()
def embed(model, x):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return F.normalize(model(x)[1].float(), dim=1)


@torch.no_grad()
def collect(model, rs, order, codec, max_per_species):
    """流式遍历测试集，只保留每段的统计量和 (物种序号, 片段) 索引。"""
    stats = {'psnr_c': [], 'psnr_r': [], 'cos_c': [], 'cos_r': [], 'label': []}
    meta = []
    rng = np.random.RandomState(0)
    for k, ph in enumerate(order):
        items = phase_items(ph, 'test', 'clip')
        if max_per_species:
            items = [items[i] for i in rng.permutation(len(items))[:max_per_species]]
        loader = DataLoader(_PairDataset(items, codec), batch_size=16, num_workers=12)
        for x, xc, y in loader:
            x, xc = x.cuda(non_blocking=True), xc.cuda(non_blocking=True)
            xr = rs[k].restore_uint8(xc)
            eo, ec, er = embed(model, x), embed(model, xc), embed(model, xr)
            stats['psnr_c'] += psnr(xc, x).tolist()
            stats['psnr_r'] += psnr(xr, x).tolist()
            stats['cos_c'] += (eo * ec).sum(1).tolist()
            stats['cos_r'] += (eo * er).sum(1).tolist()
            stats['label'] += y.tolist()
        meta += [(k, it) for it in items]
        print(f'species {ph}: {len(stats["psnr_c"])} clips so far', flush=True)
    return {k: np.array(v) for k, v in stats.items()}, meta


def rep_distance(st):
    """每段离"典型"的距离：压缩后 PSNR、PSNR 提升、余弦提升与全体中位数之差（按四分位距归一化）之和；非 Strong 为 inf。"""
    feats = [st['psnr_c'], st['psnr_r'] - st['psnr_c'], st['cos_r'] - st['cos_c']]
    d = sum(np.abs(f - np.median(f)) / (np.percentile(f, 75) - np.percentile(f, 25)) for f in feats)
    d[st['label'] != 3] = np.inf
    return d


def pick_representative(st):
    """Strong 片段中，压缩后 PSNR、PSNR 提升、余弦提升三者都最接近全体中位数的一段（按四分位距归一化）。"""
    feats = [st['psnr_c'], st['psnr_r'] - st['psnr_c'], st['cos_r'] - st['cos_c']]
    d = sum(np.abs(f - np.median(f)) / (np.percentile(f, 75) - np.percentile(f, 25)) for f in feats)
    d[st['label'] != 3] = np.inf
    return int(np.argmin(d))


@torch.no_grad()
def example_frames(i, meta, rs, codec):
    k, it = meta[i]
    x, xc, _ = _PairDataset([it], codec)[0]
    xr = rs[k].restore_uint8(xc[None].cuda())[0].cpu()
    motion = (x[FRAME + 1].float() - x[FRAME - 1].float()).abs().mean(-1)   # 原始片段相邻帧差：鱼和水花在动
    return x[FRAME], xc[FRAME], xr[FRAME], k, motion


def pick_box(motion, o=None, c=None, r=None, by='motion'):
    """by='motion'：运动最强的 BOX×BOX 区域（鱼体和水花）；by='reduction'：修复后误差下降最多的区域；by='fish'：运动区域内误差下降最多。
    避开边缘和顶部 24 px 的时间水印。"""
    if by in ('reduction', 'fish'):
        m = (c.float() - o.float()).abs().mean(-1) - (r.float() - o.float()).abs().mean(-1)
    else:
        m = motion
    pooled = F.avg_pool2d(m[None, None], BOX, stride=2)[0, 0]
    if by == 'fish':                 # 只在运动最强的 20% 区域（鱼体、水花）里找误差下降最多的位置
        pm = F.avg_pool2d(motion[None, None], BOX, stride=2)[0, 0]
        pooled[pm < pm.quantile(0.8)] = -1e9
    pooled[:12], pooled[-4:], pooled[:, :4], pooled[:, -4:] = -1e9, -1e9, -1e9, -1e9
    rr, cc = divmod(int(pooled.argmax()), pooled.shape[1])
    return rr * 2, cc * 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_per_species', type=int, default=0)
    p.add_argument('--out', default='fig_restore', help='输出文件名（不含扩展名）')
    p.add_argument('--box', type=int, default=48, help='放大区域边长（像素）')
    p.add_argument('--box_by', choices=['motion', 'reduction', 'fish'], default='motion')
    p.add_argument('--clip', default=None, help='直接指定示例片段（视频路径），否则自动选最典型的 Strong 片段')
    a = p.parse_args()
    global BOX
    BOX = a.box
    model, rs, order = load()
    codec = build_codec('h264crf33')
    cache = RESULTS / 'pilot' / 'restore_clip_stats.npz'
    if cache.exists() and not a.max_per_species:
        z = np.load(cache, allow_pickle=True)
        st = {k: z[k] for k in ('psnr_c', 'psnr_r', 'cos_c', 'cos_r', 'label')}
        meta = [(int(k), {'video': v, 'label': int(l)}) for k, v, l in zip(z['k'], z['video'], z['label'])]
    else:
        st, meta = collect(model, rs, order, codec, a.max_per_species)
        if not a.max_per_species:
            np.savez(cache, **st, k=np.array([m[0] for m in meta]), video=np.array([m[1]['video'] for m in meta]))
    idx = [m[1]['video'] for m in meta].index(a.clip) if a.clip else pick_representative(st)
    o, c, r, kk, motion = example_frames(idx, meta, rs, codec)
    ph = order[kk]
    top, left = pick_box(motion, o, c, r, a.box_by)
    ys, xs = slice(top, top + BOX), slice(left, left + BOX)

    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(11, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.34)
    ga = outer[0].subgridspec(3, 3, hspace=0.12, wspace=0.05)
    gb = outer[1].subgridspec(2, 1, hspace=0.45)

    frames = [('Original', o), ('H.264 CRF 33', c), ('Restored', r)]
    im = None
    for j, (lab, f) in enumerate(frames):
        ax = fig.add_subplot(ga[0, j])
        ax.imshow(f.numpy())
        ax.add_patch(Rectangle((left - 0.5, top - 0.5), BOX, BOX, fill=False, ec='red', lw=1.5))
        title = lab if j == 0 else f'{lab}\nPSNR {psnr(f[None, None], o[None, None]).item():.1f} dB'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        ax = fig.add_subplot(ga[1, j])
        ax.imshow(f[ys, xs].numpy(), interpolation='nearest')
        for s in ax.spines.values():
            s.set_edgecolor('red')
            s.set_linewidth(1.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(ga[2, j])
        if j == 0:                         # 第三排最左：行标签
            ax.axis('off')
            ax.text(0.5, 0.5, f'Absolute error\n(×{GAIN})', ha='center', va='center', fontsize=10, transform=ax.transAxes)
            continue
        err = ((f[ys, xs].float() - o[ys, xs].float()).abs().mean(-1) / 255 * GAIN).clamp(0, 1)
        im = ax.imshow(err.numpy(), cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'|{"Comp." if j == 1 else "Rest."} − Orig.|', fontsize=8.5)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.05, 0.0, 0.06, 1.0]))   # 竖放在右侧误差图旁
    cb.ax.tick_params(labelsize=8)
    fig.text(0.125, 0.955, '(a)', fontsize=11, fontweight='bold')

    kw = dict(alpha=0.6, edgecolor='white', linewidth=0.3)
    ax = fig.add_subplot(gb[0])
    bins = np.linspace(np.percentile(np.r_[st['psnr_c'], st['psnr_r']], 0.5), np.percentile(np.r_[st['psnr_c'], st['psnr_r']], 99.5), 40)
    ax.hist(st['psnr_c'], bins=bins, color='#4c78a8', label=f'Compressed (mean {st["psnr_c"].mean():.1f} dB)', **kw)
    ax.hist(st['psnr_r'], bins=bins, color='#f58518', label=f'Restored (mean {st["psnr_r"].mean():.1f} dB)', **kw)
    ax.set_xlabel('PSNR to original (dB)')
    ax.set_ylabel('Number of clips')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('(b)', loc='left', fontweight='bold', fontsize=11)
    ax = fig.add_subplot(gb[1])
    lo = np.percentile(np.r_[st['cos_c'], st['cos_r']], 0.5)
    bins = np.linspace(lo, 1, 40)
    ax.hist(st['cos_c'], bins=bins, color='#4c78a8', label=f'Compressed (mean {st["cos_c"].mean():.3f})', **kw)
    ax.hist(st['cos_r'], bins=bins, color='#f58518', label=f'Restored (mean {st["cos_r"].mean():.3f})', **kw)
    ax.set_xlabel('Cosine similarity to original feature')
    ax.set_ylabel('Number of clips')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('(c)', loc='left', fontweight='bold', fontsize=11)
    for ax in fig.axes[-2:]:
        ax.spines[['top', 'right']].set_visible(False)

    OUT.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{a.out}.{ext}', bbox_inches='tight', dpi=300)
    res = {'n_clips': int(len(st['psnr_c'])), 'example_species_phase': int(ph), 'box_top_left': [top, left],
           'example_clip': meta[idx][1]['video'], 'example_psnr_c': float(st['psnr_c'][idx]),
           'example_psnr_r': float(st['psnr_r'][idx]), 'example_cos_c': float(st['cos_c'][idx]), 'example_cos_r': float(st['cos_r'][idx]),
           'median_psnr_gain': float(np.median(st['psnr_r'] - st['psnr_c'])), 'median_cos_gain': float(np.median(st['cos_r'] - st['cos_c'])),
           **{f'{k}_mean': float(v.mean()) for k, v in st.items() if k != 'label'},
           'frac_cos_improved': float((st['cos_r'] > st['cos_c']).mean()),
           'frac_psnr_improved': float((st['psnr_r'] > st['psnr_c']).mean())}
    json.dump(res, open(RESULTS / 'pilot' / 'restore_fig_stats.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
