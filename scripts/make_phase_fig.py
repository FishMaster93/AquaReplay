"""
论文图：各阶段的总体准确率 (a) 和遗忘 (b)，所有方法 200 MB、最终配置。
有 3 个种子的方法（Ours、MRDC、Joint）画均值，其余为种子 42。图例放在两个子图下方，不遮挡折线。
输出 paper_figures/fig_phase_curves.{pdf,png}
  python scripts/make_phase_fig.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.paths import RESULTS, ROOT

RES = RESULTS / 'e2e' / 'clip'
OUT = ROOT / 'paper_figures'
SEEDS = (42, 43, 44)

# (图例, 结果文件前缀, 是否 3 种子, 颜色, 线型, 标记)
METHODS = [
    ('Fine-tuning', 'finetune_b200_main', False, '#7f7f7f', '--', 'v'),
    ('LwF', 'finetune_b200_lwf', False, '#ff9f1c', '--', 's'),
    ('EWC', 'finetune_b200_ewc', False, '#9467bd', ':', 'D'),
    ('SMILE', 'smile_b200_main', False, '#bcbd22', '-.', 'P'),
    ('vCLIMB', 'vclimb_b200_main', False, '#8c564b', '-.', 'p'),
    ('iCaRL', 'icarl_b200_main', False, '#2ca02c', '-.', '^'),
    ('DER++', 'der_b200_main', False, '#e377c2', '-.', 'h'),
    ('ER', 'er_b200_main', False, '#17becf', '-.', 'X'),
    ('MRDC', 'mrdc_b200_main', True, '#1f3b73', '-', 'o'),
    ('Ours', 'aquareplay_b200_main', True, '#d62728', '-', 'o'),
]


def curves(prefix, multi):
    fs = [RES / f'{prefix}_seed{s}.json' for s in (SEEDS if multi else SEEDS[:1])]
    runs = [json.load(open(f)) for f in fs if f.exists()]          # 有几个种子用几个
    acc = np.mean([[100 * h['acc_seen'] for h in r['history']] for r in runs], 0)
    fg = np.mean([[100 * h['avg_forget'] for h in r['history']] for r in runs], 0)
    return acc, fg


def main():
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.titleweight': 'bold',
                         'axes.labelweight': 'bold', 'axes.titlesize': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    phases = np.arange(7)
    handles = []
    for lab, prefix, multi, c, ls, mk in METHODS:
        acc, fg = curves(prefix, multi)
        ours = lab == 'Ours'
        kw = dict(color=c, ls=ls, marker=mk, lw=2.8 if ours else 1.4, ms=6 if ours else 5,
                  zorder=4 if ours else 2, mec='white' if ours else c, mew=0.6)
        h, = ax1.plot(phases, acc, label=lab, **kw)
        ax2.plot(phases[1:], fg[1:], **kw)
        handles.append(h)
    joint = np.mean([100 * json.load(open(f))['final_acc'] for f in (RES / f'joint_b200_main_seed{s}.json' for s in SEEDS)
                     if f.exists()])
    hj, = ax1.plot(phases, [joint] * 7, ls=':', color='#1f77b4', marker='*', ms=9, lw=1.6, label='Joint', zorder=3)
    handles.append(hj)

    ax1.set(title='(a) Accuracy across phases', xlabel='Phase', ylabel='Overall accuracy (%)', xticks=phases)
    ax2.set(title='(b) Forgetting across phases', xlabel='Phase', ylabel='Forgetting (%)', xticks=phases[1:])
    ax1.set_ylim(30, 95)
    ax2.set_ylim(0, None)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, lw=0.6)
    fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc='lower center', ncol=6,
               frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handlelength=2.6)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_phase_curves.{ext}', bbox_inches='tight', dpi=300)
    print('saved', OUT / 'fig_phase_curves.pdf')


if __name__ == '__main__':
    main()
