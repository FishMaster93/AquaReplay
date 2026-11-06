"""
论文图（修复器效果，两个场景合一张）：
  (a) 每个场景一行：整帧（红框）| 红框区域放大：原始 / H.264 CRF 33 / 修复后 | |压缩 − 原始|、|修复 − 原始|（×4，magma，0–1）
      第一行广州红罗非鱼，第二行烟台 Oplegnathus punctatus，均为 Strong 等级
  (b) 全部测试片段的 PSNR 分布；(c) 特征余弦相似度分布（读 make_restore_fig.py 生成的逐片段统计缓存）
输出 paper_figures/fig_restore.{pdf,png}
  python scripts/make_restore_fig2.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_restore_fig as M
from vcil.data import load_manifest
from vcil.e2e.codecs import build_codec
from vcil.paths import RESULTS

SCENES = [  # (行标签, 片段, 选框方式)
    ('Guangzhou\nred tilapia', 'gz/20231208AM/Red_tilapia_2/Strong/47', 'fish'),          # sample_id（见 data/manifest_video.csv）
    ('Yantai\nO. punctatus', 'opleg/2022_6_19/PM_40/strong/19_audio_21', 'reduction'),
]
M.BOX = 32
GAIN = M.GAIN


def main():
    z = np.load(RESULTS / 'pilot' / 'restore_clip_stats.npz', allow_pickle=True)
    st = {k: z[k] for k in ('psnr_c', 'psnr_r', 'cos_c', 'cos_r')}
    meta = [(int(k), {'video': v, 'label': int(l)}) for k, v, l in zip(z['k'], z['video'], z['label'])]
    videos = list(z['video'])
    model, rs, order = M.load()
    codec = build_codec('h264crf33')

    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(11, 6.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.32)
    ga = outer[0].subgridspec(2, 7, width_ratios=[0.55, 1, 1, 1, 1, 1, 1], wspace=0.06, hspace=0.12)
    B = M.BOX
    im = None
    for row, (lab, clip, by) in enumerate(SCENES):
        path = next(row['video_path'] for row in load_manifest() if row['sample_id'] == clip)
        o, c, r, _, motion = M.example_frames(videos.index(path), meta, rs, codec)
        t, l = M.pick_box(motion, o, c, r, by)
        ys, xs = slice(t, t + B), slice(l, l + B)
        ax = fig.add_subplot(ga[row, 0])
        ax.axis('off')
        ax.text(0.95, 0.5, lab, ha='right', va='center', fontsize=9.5, transform=ax.transAxes)
        ax = fig.add_subplot(ga[row, 1])
        ax.imshow(o.numpy())
        ax.add_patch(Rectangle((l - 0.5, t - 0.5), B, B, fill=False, ec='red', lw=1.3))
        ax.axis('off')
        if row == 0:
            ax.set_title('Frame', fontsize=9)
        for j, (name, f) in enumerate([('Original', o), ('H.264 CRF 33', c), ('Restored', r)]):
            ax = fig.add_subplot(ga[row, 2 + j])
            ax.imshow(f[ys, xs].numpy(), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('red'); s.set_linewidth(1.3)
            p = M.psnr(f[None, None], o[None, None]).item()
            if row == 0:
                ax.set_title(name, fontsize=9)
            if j:                      # PSNR 写在放大图右下角，避免和下一行重叠
                ax.text(0.97, 0.04, f'{p:.1f} dB', ha='right', va='bottom', fontsize=8, color='white',
                        transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, lw=0))
        for j, f in enumerate((c, r)):
            ax = fig.add_subplot(ga[row, 5 + j])
            err = ((f[ys, xs].float() - o[ys, xs].float()).abs().mean(-1) / 255 * GAIN).clamp(0, 1)
            im = ax.imshow(err.numpy(), cmap='magma', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'|{"Comp." if j == 0 else "Rest."} − Orig.| ×{GAIN}', fontsize=9)
        if row == 1:
            last = ax
    cb = fig.colorbar(im, cax=last.inset_axes([1.06, 0.0, 0.07, 2.12]))
    cb.ax.tick_params(labelsize=8)
    fig.text(0.1, 0.905, '(a)', fontsize=11, fontweight='bold')

    gb = outer[1].subgridspec(1, 2, wspace=0.28)
    kw = dict(alpha=0.6, edgecolor='white', linewidth=0.3)
    for j, (key, xl, fmt, unit) in enumerate([('psnr', 'PSNR to original (dB)', '{:.1f}', ' dB'),
                                              ('cos', 'Cosine similarity to original feature', '{:.3f}', '')]):
        ax = fig.add_subplot(gb[j])
        a_c, a_r = st[f'{key}_c'], st[f'{key}_r']
        both = np.r_[a_c, a_r]
        bins = np.linspace(np.percentile(both, 0.5), np.percentile(both, 99.5) if key == 'psnr' else 1, 40)
        ax.hist(a_c, bins=bins, color='#4c78a8', label=f'Compressed (mean {fmt.format(a_c.mean())}{unit})', **kw)
        ax.hist(a_r, bins=bins, color='#f58518', label=f'Restored (mean {fmt.format(a_r.mean())}{unit})', **kw)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.set_xlabel(xl)
        ax.set_ylabel('Number of clips')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        ax.set_title('(b)' if j == 0 else '(c)', loc='left', fontweight='bold', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)

    for ext in ('pdf', 'png'):
        fig.savefig(M.OUT / f'fig_restore.{ext}', bbox_inches='tight', dpi=300)
    print('saved', M.OUT / 'fig_restore.pdf')


if __name__ == '__main__':
    main()
