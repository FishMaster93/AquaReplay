"""
纯视频物种增量 FFIR 的数据定义。所有划分都来自 data/manifest_video.csv（由 scripts/build_manifest.py 生成）：

  protocol='clip'      : 片段级 80/10/10，与 AquaLytic 音频实验逐样本相同（列 split_clip），
                         便于和 AquaLytic 的音频结果直接对照
  protocol='session'   : 按天切分（同一天的 AM/PM 场次整体分到同一子集），列 split_s0；
           'session{k}': 列 split_s{k}，k=0..4

样本以视频 npy 路径为唯一 key；每段视频是预处理好的 (16, 224, 224, 3) uint8 张量。
"""
import csv
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from vcil.paths import MANIFEST

PHASES = {
    0: ['Red_tilapia_2', 'Red_tilapia_3'],
    1: ['Tilapia'],
    2: ['Jade_perch'],
    3: ['Largemouth_bass'],
    4: ['Lotus_carp'],
    5: ['Sunfish'],
    6: ['Oplegnathus_punctatus'],
}
CLASS_NAMES = ['None', 'Weak', 'Medium', 'Strong']
C = len(CLASS_NAMES)


def split_col(protocol):
    if protocol == 'clip':
        return 'split_clip'
    if protocol.startswith('session'):
        return f'split_s{protocol[len("session"):] or "0"}'
    raise ValueError(f'未知协议: {protocol}')


@lru_cache(maxsize=1)
def load_manifest():
    with open(MANIFEST, newline='') as f:
        return list(csv.DictReader(f))


def phase_items(phase, split=None, protocol='clip'):
    """某阶段（若干物种）的样本列表；split=None 返回该阶段全部视频（抽特征用）。"""
    species = set(PHASES[phase])
    col = split_col(protocol) if split else None
    return [{'video': r['video_path'], 'label': int(r['label']), 'species': r['species'], 'session': r['session']}
            for r in load_manifest()
            if r['species'] in species and (split is None or r[col] == split)]


class VideoClipDataset(Dataset):
    """读预处理好的 (16, 224, 224, 3) uint8 视频张量。增强放到 GPU 上做。"""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        clip = torch.from_numpy(np.load(it['video']))
        return clip, it['label'], i


def gpu_augment(x, *paired, scale=(0.8, 1.0)):
    """x: (B,T,H,W,3) uint8 on GPU -> 同形状 uint8。每个样本独立：随机水平翻转 + 随机裁剪后缩放回原尺寸。
    paired 里的张量（如同一批片段的压缩版）与 x 用完全相同的增强参数，返回 (x', *paired')。"""
    B, T, H, W, _ = x.shape
    outs = [torch.empty_like(t) for t in (x, *paired)]
    for i in range(B):
        s = np.random.uniform(*scale)
        h, w = int(H * s), int(W * s)
        top, left = np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
        flip = np.random.rand() < 0.5
        for src, out in zip((x, *paired), outs):
            clip = src[i][:, top:top + h, left:left + w, :]
            if flip:
                clip = clip.flip(2)
            clip = clip.permute(0, 3, 1, 2).float()
            clip = F.interpolate(clip, size=(H, W), mode='bilinear', align_corners=False)
            out[i] = clip.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return outs[0] if not paired else tuple(outs)
