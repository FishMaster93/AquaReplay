"""
论文图：每个阶段后各物种的准确率热力图（AquaReplay / MRDC / 两者之差，种子 42）。输出 paper_figures/fig_species_mat.{pdf,png}
  python scripts/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.data import PHASES
from vcil.paths import RESULTS, ROOT

RES = RESULTS / 'e2e' / 'clip'
OUT = ROOT / 'paper_figures'
SPECIES = ['Red tilapia', 'Tilapia', 'Jade perch', 'Largemouth bass', 'Lotus carp', 'Sunfish', 'Oplegnathus']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def load(*names):
    """依次尝试 names，返回 (结果, 是否为旧配置)。"""
    for i, n in enumerate(names):
        f = RES / f'{n}.json'
        if f.exists():
            return json.load(open(f)), i > 0
    return None, None


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('saved', OUT / f'{name}.pdf')


def species_mat():
    """(a) AquaReplay、(b) MRDC：每个阶段后各物种的准确率；(c) 两者之差。种子 42，默认顺序。"""
    runs = [('(a) AquaReplay', 'aquareplay_b200_main_seed42'), ('(b) MRDC', 'mrdc_b200_main_seed42')]
    names = SPECIES[:-1] + ['$\\it{O.\\ punctatus}$']
    mats = []
    for _, n in runs:
        d, _ = load(n)
        order = d['order']
        m = np.full((len(order), len(order)), np.nan)
        for i, h in enumerate(d['history']):
            for j, ph in enumerate(order[:i + 1]):
                m[i, j] = 100 * h['per_phase'][str(ph)]
        mats.append(m)
    panels = runs + [('(c) AquaReplay − MRDC', None)]
    fig = plt.figure(figsize=(12.5, 3.8))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.045, 0.32, 1, 0.045], wspace=0.08)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[4])]
    caxes = {'(b)': fig.add_subplot(gs[2]), '(c)': fig.add_subplot(gs[5])}
    for ax, (lab, _), m in zip(axes, panels, mats + [mats[0] - mats[1]]):
        diff = lab.startswith('(c)')
        im = ax.imshow(m, cmap='RdBu' if diff else 'viridis', vmin=-15 if diff else 65, vmax=15 if diff else 95)
        for i in range(m.shape[0]):
            for j in range(i + 1):
                v = m[i, j]
                dark = abs(v) > 9 if diff else v < 80
                txt = (f'{v:+.0f}' if round(v) else '0') if diff else f'{v:.0f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=6.5,
                        color='white' if dark else 'black')
        ax.set_title(lab, fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f'after {n}' for n in names] if lab.startswith('(a)') else [], fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        if lab.startswith('(b)') or diff:   # (a)(b) 共用一个色标，放在 (b) 右侧；(c) 单独一个
            cb = fig.colorbar(im, cax=caxes[lab[:3]])
            cb.set_label('Accuracy difference (pp)' if diff else 'Accuracy (%)', fontsize=8)
            cb.ax.tick_params(labelsize=7)
    save(fig, 'fig_species_mat')


if __name__ == '__main__':
    OUT = ROOT / 'paper_figures'
    species_mat()
