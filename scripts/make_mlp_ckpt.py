"""
把 Phase 0 checkpoint 的线性头换成 MLP CORAL 头：骨干不动，MLP 头在 Phase 0 训练集的冻结特征上训练
（Adam lr 1e-3，30 轮，batch 256），Phase 0 测试集上报告准确率。输出 workspace/S3D_phase0_clip/phase0_mlp.pt。
端到端增量学习从这个 checkpoint 开始（--ckpt）。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.models.video import VideoModel, coral_loss, coral_predict
from vcil.paths import WORKSPACE


@torch.no_grad()
def embed_items(model, items):
    """在冻结骨干上现算特征（非 S3D 骨干没有预先提取的特征文件）。"""
    from torch.utils.data import DataLoader
    from vcil.data import VideoClipDataset
    xs, ys = [], []
    model.eval()
    for clip, y, _ in DataLoader(VideoClipDataset(items), batch_size=32, num_workers=8, pin_memory=True):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            xs.append(model.backbone(clip.cuda(non_blocking=True))['embedding'].float())
        ys.append(y.cuda())
    return torch.cat(xs), torch.cat(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=str(WORKSPACE / 'S3D_phase0_clip' / 'phase0_best.pt'))
    p.add_argument('--out', default=str(WORKSPACE / 'S3D_phase0_clip' / 'phase0_mlp.pt'))
    a = p.parse_args()
    src = a.src
    state = torch.load(src, map_location='cuda', weights_only=False)
    backbone = state.get('backbone', 's3d')
    model = VideoModel(mlp=True, backbone=backbone).cuda()
    model.backbone.load_state_dict({k[len('backbone.'):]: v for k, v in state['model_state_dict'].items()
                                    if k.startswith('backbone.')})
    from vcil.data import phase_items
    x, y = embed_items(model, phase_items(0, 'train', 'clip'))
    xt, yt = embed_items(model, phase_items(0, 'test', 'clip'))
    torch.manual_seed(0)
    head = model.head
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(30):
        head.train()
        perm = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), 256):
            j = perm[i:i + 256]
            loss = coral_loss(head(x[j]), y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        acc = float((coral_predict(head(xt)) == yt).float().mean())
    print(f'Phase 0 test acc with MLP head on frozen features: {acc:.4f}')
    torch.save({'model_state_dict': model.state_dict(), 'mlp': True, 'backbone': backbone,
                'phase0_test_acc': acc, 'src': str(src)}, a.out)
    print(f'saved -> {a.out}')


if __name__ == '__main__':
    main()
