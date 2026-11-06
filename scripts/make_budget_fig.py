"""
论文图 4：(a) 最终准确率 vs 记忆预算（AquaReplay / MRDC，200 MB 处为 3 个种子均值 ± 样本标准差，联合训练为上限虚线）；
(b) 最终准确率 vs 存储片段数（各预算下的 AquaReplay、MRDC 曲线 + 200 MB 下的其他存样本方法）。
冻结编码器只在 5.5 节讨论，不画在这里。所有标注都在坐标区内。
输出 paper_figures/fig_budget.{pdf,png}
  python scripts/make_budget_fig.py
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
C_OURS, C_MRDC = '#2b6fd6', '#e8692e'


def point(prefix):
    """返回 (片段数, 均值, 样本标准差或 None)；有 3 个种子就取均值。"""
    fs = [RES / f'{prefix}_seed{s}.json' for s in SEEDS]
    runs = [json.load(open(f)) for f in fs if f.exists()]
    acc = [100 * r['final_acc'] for r in runs]
    n = int(np.mean([r['memory']['n_exemplars'] for r in runs]))
    return n, float(np.mean(acc)), (float(np.std(acc, ddof=1)) if len(acc) > 1 else None)


def series(fmt, budgets):
    return [(b, *point(fmt.format(b))) for b in budgets]


def main():
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    ours = series('aquareplay_b{}_main', [20, 50, 200, 500, 1000])
    mrdc = series('mrdc_b{}_main', [50, 200, 500, 1000])
    joint = np.mean([100 * json.load(open(RES / f'joint_b200_main_seed{s}.json'))['final_acc'] for s in SEEDS])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))

    # (a) 准确率 vs 预算
    for lab, pts, c, mk in [('AquaReplay', ours, C_OURS, 'o'), ('MRDC', mrdc, C_MRDC, 's')]:
        b = [p[0] for p in pts]
        a = [p[2] for p in pts]
        e = [p[3] or 0 for p in pts]
        ax1.errorbar(b, a, yerr=e, color=c, marker=mk, ms=6, lw=2, capsize=3, label=lab)
    ax1.axhline(joint, ls=':', color='0.35', lw=1.2)
    ax1.text(1000, joint + 0.35, 'Joint training', ha='right', va='bottom', fontsize=9, color='0.35')
    ax1.set_xscale('log')
    ax1.set_xticks([20, 50, 200, 500, 1000])
    ax1.set_xticklabels(['20', '50', '200', '500', '1000'])
    ax1.minorticks_off()
    ax1.set_xlim(15, 1300)
    ax1.set_ylim(72, 90)
    ax1.set_xlabel('Memory budget (MB, log scale)')
    ax1.set_ylabel('Final accuracy (%)')
    ax1.set_title('(a) Final accuracy vs. memory budget')
    ax1.legend(frameon=False, loc='center left', bbox_to_anchor=(0.0, 0.62))
    ax1.grid(alpha=0.3, lw=0.6)

    # (b) 准确率 vs 存储片段数
    for lab, pts, c, mk in [('AquaReplay', ours, C_OURS, 'o'), ('MRDC', mrdc, C_MRDC, 's')]:
        ax2.plot([p[1] for p in pts], [p[2] for p in pts], color=c, marker=mk, ms=5, lw=2, zorder=3)
        n200, a200 = next((p[1], p[2]) for p in pts if p[0] == 200)
        ax2.scatter([n200], [a200], s=170, facecolors='none', edgecolors='0.2', lw=1.3, zorder=4)
    ax2.annotate('AquaReplay', (ours[-1][1], ours[-1][2]), xytext=(-6, 9), textcoords='offset points',
                 ha='right', color=C_OURS, fontsize=9.5, fontweight='bold')
    ax2.annotate('200 MB', (ours[2][1], ours[2][2]), xytext=(4, -15), textcoords='offset points', fontsize=8.5, color='0.25')
    ax2.annotate('MRDC', (mrdc[0][1], mrdc[0][2]), xytext=(2, -15), textcoords='offset points',
                 color=C_MRDC, fontsize=9.5, fontweight='bold')
    ax2.annotate('200 MB', (mrdc[1][1], mrdc[1][2]), xytext=(4, -15), textcoords='offset points', fontsize=8.5, color='0.25')

    others = [('ER', 'er_b200_main', '#6a3fc8', 'v'), ('DER++', 'der_b200_main', '#2a8a3e', 'X'),
              ('iCaRL', 'icarl_b200_main', '#e0a000', 'D'), ('vCLIMB', 'vclimb_b200_main', '#e5508f', 'P'),
              ('SMILE', 'smile_b200_main', '#1aa39a', '^')]
    for lab, prefix, c, mk in others:
        n, a, _ = point(prefix)
        ax2.scatter([n], [a], color=c, marker=mk, s=48, zorder=3)
        ax2.annotate(lab, (n, a), xytext=(7, -3), textcoords='offset points', fontsize=9, va='center')
    ax2.set_xscale('log')
    ax2.set_xlim(30, 1.5e5)
    ax2.set_ylim(40, 90)
    ax2.set_xlabel('Number of stored clips (log scale)')
    ax2.set_ylabel('Final accuracy (%)')
    ax2.set_title('(b) Final accuracy vs. number of stored clips')
    ax2.grid(alpha=0.3, lw=0.6)

    fig.tight_layout(w_pad=2.5)
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_budget.{ext}', bbox_inches='tight', dpi=300)
    print('saved', OUT / 'fig_budget.pdf')
    for lab, pts in [('AquaReplay', ours), ('MRDC', mrdc)]:
        print(lab, [(b, n, round(a, 2), None if e is None else round(e, 2)) for b, n, a, e in pts])


if __name__ == '__main__':
    main()
