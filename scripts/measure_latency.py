"""
推理延迟：一段 16 帧 224×224 片段（batch 1）经过骨干 + MLP 序数头的时间。
GPU 测 fp32 和 bf16 两种（各预热 20 次、计时 100 次取中位数）；CPU 用 fp32、全部线程（预热 3 次、计时 20 次取中位数）。
延迟只取决于网络结构，所以 R(2+1)D-18、MViTv2-S 直接用 torchvision 随机初始化的结构测。
结果写 results/pilot/latency.json
  python scripts/measure_latency.py
"""
import json
import os
import platform
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.models.video import VideoModel
from vcil.paths import RESULTS


class TVModel(nn.Module):
    """R(2+1)D-18 / MViTv2-S：torchvision 结构 + 与 VideoModel 相同的 MLP 序数头。"""

    def __init__(self, name):
        super().__init__()
        from torchvision.models import video as V
        net = getattr(V, name)(weights=None)
        if name == 'mvit_v2_s':
            d = net.head[-1].in_features
            net.head = nn.Identity()
        else:
            d = net.fc.in_features
            net.fc = nn.Identity()
        self.net = net
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 3))

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).float().div(255.0)
        return self.head(self.net(x))


def build(name):
    if name in ('s3d', 'x3d_m', 'mc3_18'):
        return VideoModel(mlp=True, backbone=name)
    return TVModel(name)


@torch.no_grad()
def timeit(model, x, n_warm, n_rep, sync, autocast=None):
    def run():
        if autocast:
            with torch.autocast('cuda', dtype=autocast):
                model(x)
        else:
            model(x)
    for _ in range(n_warm):
        run()
    ts = []
    for _ in range(n_rep):
        sync()
        t0 = time.perf_counter()
        run()
        sync()
        ts.append(1000 * (time.perf_counter() - t0))
    return statistics.median(ts)


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(os.cpu_count())
    names = ['s3d', 'x3d_m', 'mc3_18', 'r2plus1d_18', 'mvit_v2_s']
    x = torch.randint(0, 256, (1, 16, 224, 224, 3), dtype=torch.uint8)
    res = {'gpu': torch.cuda.get_device_name(0), 'cpu': platform.processor() or platform.machine(),
           'cpu_threads': torch.get_num_threads(), 'input': '1 x 16 x 224 x 224 x 3 uint8', 'models': {}}
    for name in names:
        m = build(name).eval()
        params = sum(p.numel() for p in m.parameters()) / 1e6
        mg = m.cuda()
        xg = x.cuda()
        sync = torch.cuda.synchronize
        g32 = timeit(mg, xg, 20, 100, sync)
        g16 = timeit(mg, xg, 20, 100, sync, torch.bfloat16)
        mc = m.cpu()
        c32 = timeit(mc, x, 3, 20, lambda: None)
        res['models'][name] = {'params_M': round(params, 2), 'gpu_fp32_ms': round(g32, 2), 'gpu_bf16_ms': round(g16, 2),
                               'cpu_fp32_ms': round(c32, 1)}
        print(name, res['models'][name], flush=True)
        del m, mg, mc
        torch.cuda.empty_cache()
    json.dump(res, open(RESULTS / 'pilot' / 'latency.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'models'}))


if __name__ == '__main__':
    main()
