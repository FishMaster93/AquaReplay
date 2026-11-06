"""
端到端有样本物种增量实验驱动（编码器不冻结，记忆存视频，按 MB 计预算）。
指标定义与 vcil/cil.py 相同（acc_seen / avg_inc_acc / avg_forget / zero_shot）。
结果：results/e2e/<protocol>/<method>_b<budget>[_tag]_seed<s>.json，日志在 logs/e2e/。

  python scripts/run_e2e.py --method aquareplay --ckpt workspace/S3D_phase0_clip/phase0_mlp.pt \
      --restore workspace/restorer/restorer_task0.2.pt --budget_mb 200
  python scripts/run_e2e.py --method joint            # 上界：在全部 7 个物种上一次性训练，只报告最终准确率
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.data import PHASES, phase_items
from vcil.e2e.learner import METHODS, E2ELearner
from vcil.paths import RESULTS, ROOT, WORKSPACE


def default_order(order=None):
    """Phase 0 (red tilapia) always comes first; `order` gives the order of phases 1-6."""
    order = [0] + (list(order) if order else [p for p in PHASES if p != 0])
    assert sorted(order) == sorted(PHASES), order
    return order


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=list(METHODS) + ['joint'])
    p.add_argument('--protocol', default='clip')
    p.add_argument('--ckpt', default=str(WORKSPACE / 'S3D_phase0_clip' / 'phase0_best.pt'))
    p.add_argument('--budget_mb', type=float, default=200)
    p.add_argument('--epochs', type=int, default=3, help='每个阶段的训练轮数')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--bs', type=int, default=12)
    p.add_argument('--lam', type=float, default=1.0, help='压缩一致性权重 λ')
    p.add_argument('--der_alpha', type=float, default=0.5, help='DER++ logits 蒸馏权重 α')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--order', type=int, nargs='*', default=None)
    p.add_argument('--codec', default=None, help='覆盖方法默认的存储格式，如 h264crf28')
    p.add_argument('--select', default=None, choices=[None, 'random', 'kcenter', 'boundary', 'herding'])
    p.add_argument('--consistency', type=int, default=None, choices=[None, 0, 1])
    p.add_argument('--cons_type', default=None, choices=[None, 'logit', 'prob', 'feat'])
    p.add_argument('--cons_cls', type=int, default=None, choices=[None, 0, 1])
    p.add_argument('--anchor', type=int, default=None, choices=[None, 0, 1])
    p.add_argument('--mu', type=float, default=1.0, help='原始特征锚点权重 μ')
    p.add_argument('--restore', default=None, help='修复网络权重路径')
    p.add_argument('--restore_per_species', type=int, default=None, choices=[None, 0, 1])
    p.add_argument('--restore_steps', type=int, default=None)
    p.add_argument('--restore_w', type=float, default=None)
    p.add_argument('--tune_from', type=int, default=None, help='只微调 S3D 第 N 个模块及之后（14 = Mixed_5b）')
    p.add_argument('--refit_epochs', type=int, default=None, help='每阶段在特征上重拟合分类头的轮数')
    p.add_argument('--refit_balance', type=int, default=None, choices=[None, 0, 1])
    p.add_argument('--align', type=float, default=None, help='强度对齐的跨物种迁移损失权重')
    p.add_argument('--align_tau', type=float, default=None)
    p.add_argument('--kd_feat', type=float, default=None)
    p.add_argument('--kd_out', type=float, default=None)
    p.add_argument('--ewc', type=float, default=None)
    p.add_argument('--alloc', choices=['cell', 'species', 'prop'], default=None, help='记忆字节配额的分配方式（默认 cell）')
    p.add_argument('--save_ckpt', type=int, default=0, help='1 = 结束时保存模型和各物种修复网络（画 t-SNE / 可视化用）')
    p.add_argument('--sp_thresh', type=int, default=None, choices=[None, 0, 1, 2])
    p.add_argument('--freeze_bn', type=int, default=None, choices=[None, 0, 1])
    p.add_argument('--max_phases', type=int, default=None, help='快速筛选：只跑前 N 个阶段')
    p.add_argument('--tag', default='')
    p.add_argument('--max_items', type=int, default=None, help='冒烟测试：每阶段只取前 N 个训练/测试样本')
    args = p.parse_args()

    order = default_order(args.order)[:args.max_phases]
    name = f'{args.method}_b{args.budget_mb:g}{"_" + args.tag if args.tag else ""}_seed{args.seed}'
    out = RESULTS / 'e2e' / args.protocol / f'{name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs' / 'e2e').mkdir(parents=True, exist_ok=True)
    logf = open(ROOT / 'logs' / 'e2e' / f'{args.protocol}_{name}.log', 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    log(f'args: {vars(args)}')
    train = {ph: phase_items(ph, 'train', args.protocol) for ph in order}
    test = {ph: phase_items(ph, 'test', args.protocol) for ph in order}
    if args.max_items:
        rng = np.random.RandomState(0)
        train, test = ({ph: [v[i] for i in rng.permutation(len(v))[:args.max_items]] for ph, v in d.items()}
                       for d in (train, test))
    override = {k: v for k, v in (('codec', args.codec), ('select', args.select),
                                  ('consistency', None if args.consistency is None else bool(args.consistency)),
                                  ('cons_type', args.cons_type),
                                  ('cons_cls', None if args.cons_cls is None else bool(args.cons_cls)),
                                  ('anchor', None if args.anchor is None else bool(args.anchor)),
                                  ('restore', args.restore),
                                  ('restore_per_species', None if args.restore_per_species is None
                                   else bool(args.restore_per_species)),
                                  ('restore_steps', args.restore_steps), ('restore_w', args.restore_w),
                                  ('freeze_bn', None if args.freeze_bn is None else bool(args.freeze_bn)),
                                  ('tune_from', args.tune_from), ('refit_epochs', args.refit_epochs),
                                  ('align', args.align),
                                  ('refit_balance', None if args.refit_balance is None else bool(args.refit_balance)), ('align_tau', args.align_tau), ('sp_thresh', args.sp_thresh), ('kd_feat', args.kd_feat), ('kd_out', args.kd_out), ('ewc', args.ewc), ('alloc', args.alloc))
                if v is not None}
    method = 'finetune' if args.method == 'joint' else args.method
    learner = E2ELearner(method, args.ckpt, args.budget_mb, args.epochs, args.lr, args.bs, args.lam,
                         seed=args.seed, mu=args.mu, der_alpha=args.der_alpha, **override)
    log(f'config: {learner.cfg}')

    def evaluate(phases):
        accs = {}
        for q in phases:
            pred = learner.predict(test[q], order.index(q))
            y = torch.tensor([it['label'] for it in test[q]])
            accs[q] = (int((pred == y).sum()), len(y))
        return accs

    t0 = time.time()
    history, best = [], {}
    if args.method == 'joint':
        all_items = [it for ph in order for it in train[ph]]
        if args.epochs > 0:
            learner._train(all_items, log)
        if learner.cfg['refit_epochs']:
            learner.refit_head(all_items, log=log)
        accs = evaluate(order)
        acc_seen = sum(c for c, _ in accs.values()) / sum(n for _, n in accs.values())
        history.append({'step': len(order) - 1, 'acc_seen': acc_seen, 'avg_forget': 0.0, 'zero_shot': None,
                        'per_phase': {q: c / n for q, (c, n) in accs.items()}})
    else:
        for step, ph in enumerate(order):
            zs = None
            if step > 0:
                c, n = evaluate([ph])[ph]
                zs = c / n
            log(f'== step {step} phase {ph} {PHASES[ph]} ({len(train[ph])} train clips)')
            learner.learn(step, train[ph], log)
            accs = evaluate(order[:step + 1])
            per = {q: c / n for q, (c, n) in accs.items()}
            for q, a in per.items():
                best[q] = max(best.get(q, 0.0), a)
            fg = float(np.mean([best[q] - per[q] for q in order[:step]])) if step else 0.0
            acc_seen = sum(c for c, _ in accs.values()) / sum(n for _, n in accs.values())
            history.append({'step': step, 'phase': ph, 'species': PHASES[ph], 'acc_seen': acc_seen,
                            'avg_forget': fg, 'zero_shot': zs, 'per_phase': per})
            log(f'  acc_seen {acc_seen:.4f} forget {fg:.4f} zero-shot {zs} | ' +
                ' '.join(f'{q}:{a:.3f}' for q, a in per.items()) + f' | {(time.time() - t0) / 60:.1f} min')

    res = {'method': args.method, 'config': learner.cfg, 'args': vars(args), 'order': order,
           'final_acc': history[-1]['acc_seen'], 'avg_inc_acc': float(np.mean([h['acc_seen'] for h in history])),
           'final_avg_forget': history[-1]['avg_forget'], 'history': history,
           'memory': learner.memory.stats() if learner.memory is not None else None,
           'time_min': (time.time() - t0) / 60}
    json.dump(res, open(out, 'w'), indent=2)
    if args.save_ckpt:
        ck = ROOT / 'workspace' / 'e2e_ckpt' / f'{name}.pt'
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': learner.model.state_dict(), 'mlp': True, 'order': order,
                    'restorers': {k: r.state_dict() for k, r in learner.restorers.items()}}, ck)
        log(f'saved checkpoint -> {ck}')
    log(f'SUMMARY final {100 * res["final_acc"]:.2f} avg_inc {100 * res["avg_inc_acc"]:.2f} '
        f'forget {100 * res["final_avg_forget"]:.2f} | {res["time_min"]:.1f} min -> {out}')


if __name__ == '__main__':
    main()
