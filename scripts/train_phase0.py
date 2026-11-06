"""
Phase 0 视频编码器适应：S3D（Kinetics-400 初始化）+ 临时 CORAL 序数头，只在第一个阶段的物种
（Red_tilapia_2 + Red_tilapia_3）上梯度训练，与 AquaLytic 音频 Phase 0 的做法对齐：
同样的 CORAL 头和损失、Adam lr=1e-4、按验证集挑最佳模型。之后骨干冻结，临时头丢弃。

数据增强在 GPU 上做（随机水平翻转 + 轻度随机缩放裁剪）。
产物：workspace/<exp_name>/{phase0_best.pt, results.json, train-*.log}
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.data import CLASS_NAMES, VideoClipDataset, gpu_augment, phase_items
from vcil.models.video import VideoModel, coral_predict, make_loss_and_predict
from vcil.paths import WORKSPACE


@torch.no_grad()
def evaluate(model, loader, device, predict=coral_predict):
    model.eval()
    preds, targets = [], []
    for clip, label, _ in loader:
        clip = clip.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(clip)
        preds.append(predict(logits.float()).cpu())
        targets.append(label)
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    acc = float((preds == targets).mean())
    class_acc = {CLASS_NAMES[k]: float(cm[k, k] / max(cm[k].sum(), 1)) for k in range(4)}
    return {'acc': acc, 'class_acc': class_acc, 'cm': cm.tolist(), 'preds': preds, 'targets': targets}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', default='S3D_phase0_clip')
    p.add_argument('--protocol', default='clip', help='clip / session / session{k}')
    p.add_argument('--phase', type=int, default=0, help='在哪个阶段的物种上适应编码器；默认 0 = Red Tilapia')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=0, help='训练随机种子（不影响数据划分）')
    p.add_argument('--no_augment', action='store_true')
    p.add_argument('--head', choices=['coral', 'ce'], default='coral')
    p.add_argument('--backbone', choices=['s3d', 'x3d_m', 'mc3_18'], default='s3d')
    args = p.parse_args()

    out_dir = WORKSPACE / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(out_dir / f'train-{time.time():.0f}.log'),
                                  logging.StreamHandler()])
    log = logging.getLogger()
    log.info(f'args: {vars(args)}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'

    train, val, test = (phase_items(args.phase, s, args.protocol) for s in ('train', 'val', 'test'))
    log.info(f'Phase {args.phase} ({args.protocol}) 视频样本 train/val/test = {len(train)}/{len(val)}/{len(test)}')

    def loader(items, shuffle):
        return DataLoader(VideoClipDataset(items), batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True, drop_last=shuffle,
                          persistent_workers=True, prefetch_factor=4)
    train_loader, val_loader, test_loader = loader(train, True), loader(val, False), loader(test, False)

    model = VideoModel(head=args.head, backbone=args.backbone).to(device)
    loss_fn, predict = make_loss_and_predict(args.head)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_acc, best_path = -1.0, out_dir / 'phase0_best.pt'
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0, tot_loss, n_correct, n_seen = time.time(), 0.0, 0, 0
        for step, (clip, label, _) in enumerate(train_loader):
            clip, label = clip.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if not args.no_augment:
                clip = gpu_augment(clip)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(clip)
            loss = loss_fn(logits.float(), label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item() * len(label)
            n_correct += (predict(logits.detach().float()) == label).sum().item()
            n_seen += len(label)
            if step % 100 == 0:
                log.info(f'  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}')
        val_stats = evaluate(model, val_loader, device, predict)
        rec = {'epoch': epoch, 'train_loss': tot_loss / n_seen, 'train_acc': n_correct / n_seen,
               'val_acc': val_stats['acc'], 'val_class_acc': val_stats['class_acc'],
               'time_min': (time.time() - t0) / 60}
        history.append(rec)
        log.info(f'Epoch {epoch}: train loss {rec["train_loss"]:.4f} acc {rec["train_acc"]:.4f} | '
                 f'val acc {val_stats["acc"]:.4f} {val_stats["class_acc"]} | {rec["time_min"]:.1f} min')
        if val_stats['acc'] > best_acc:
            best_acc = val_stats['acc']
            torch.save({'model_state_dict': model.state_dict(), 'backbone': args.backbone, 'epoch': epoch, 'val_acc': best_acc,
                        'args': vars(args)}, best_path)
            log.info(f'  saved best -> {best_path} (val acc {best_acc:.4f})')

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_stats = evaluate(model, test_loader, device, predict)
    by_species = {}
    for sp in sorted({it['species'] for it in test}):
        mask = np.array([it['species'] == sp for it in test])
        by_species[sp] = float((test_stats['preds'][mask] == test_stats['targets'][mask]).mean())
    log.info(f'== Phase {args.phase} Test (best epoch {ckpt["epoch"]}, val {ckpt["val_acc"]:.4f}) ==')
    log.info(f'  overall acc {test_stats["acc"]:.4f}')
    log.info(f'  per-class {test_stats["class_acc"]}')
    log.info(f'  per-species {by_species}')
    log.info(f'  confusion matrix (行=真实, 列=预测, None/Weak/Medium/Strong):\n{np.array(test_stats["cm"])}')
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({'args': vars(args), 'history': history, 'best_epoch': ckpt['epoch'], 'best_val_acc': ckpt['val_acc'],
                   'test_acc': test_stats['acc'], 'test_class_acc': test_stats['class_acc'],
                   'test_by_species': by_species, 'test_cm': test_stats['cm']}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
