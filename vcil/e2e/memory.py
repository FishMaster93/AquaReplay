"""
按字节计预算的视频回放记忆。

  总预算 budget_mb 在所有已见的 (物种, 强度) 单元间平分；新物种到来时旧单元按新配额从尾部裁剪
  （每个单元内的样本按选样优先级排序，裁掉的是优先级最低的）。
  选样：random（随机顺序）/ kcenter（在模型特征空间做 k-center 贪心，覆盖度优先、避开近重复）/
        boundary（强度感知：按 CORAL 相邻阈值的不确定性挑边界样本，与靠近单元中心的代表样本交替，
        并跳过与已选样本特征余弦相似度 > dedup 的近重复片段，见 boundary_order）。
  每个样本存 (payload, 字节数, 标签, 插入时模型的 logits, 插入时原始片段的特征)；logits 供 DER++ 使用，
  特征是原始特征锚点（anchor=True 时每个样本另计 fp16 1024 维 = 2 KB 预算）。
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from vcil.data import C

MB = 1024 * 1024


class _EncodeDataset(Dataset):
    def __init__(self, items, codec):
        self.items, self.codec = items, codec

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if getattr(self.codec, 'by_ref', False):
            return self.codec.encode_ref(self.items[i]['video'])
        clip = torch.from_numpy(np.load(self.items[i]['video']))
        return self.codec.encode(clip)


class _ReplayDataset(Dataset):
    def __init__(self, entries, codec):
        self.entries, self.codec = entries, codec      # [(物种序号 k, entry), ...]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        k, (p, _, y, z, a) = self.entries[i]
        return self.codec.decode(p), y, z, a.float(), k


def herding_order(emb, n):
    """iCaRL herding：每次选使已选样本均值最接近全体均值的样本，返回前 n 个的下标顺序。"""
    emb = F.normalize(emb, dim=1)
    mu = emb.mean(0)
    order, acc = [], torch.zeros_like(mu)
    avail = torch.ones(len(emb), dtype=torch.bool, device=emb.device)
    for t in range(1, min(n, len(emb)) + 1):
        d = ((mu - (acc + emb) / t) ** 2).sum(1)
        d[~avail] = float('inf')
        i = int(d.argmin())
        order.append(i)
        acc += emb[i]
        avail[i] = False
    return order


def kcenter_order(emb, n):
    """k-center 贪心：从离均值最近的样本开始，每次选离已选集合最远的样本，返回前 n 个的下标顺序。"""
    emb = F.normalize(emb, dim=1)
    first = int(((emb - emb.mean(0)) ** 2).sum(1).argmin())
    order = [first]
    dist = ((emb - emb[first]) ** 2).sum(1)
    for _ in range(min(n, len(emb)) - 1):
        i = int(dist.argmax())
        order.append(i)
        dist = torch.minimum(dist, ((emb - emb[i]) ** 2).sum(1))
    return order


def boundary_order(emb, logits, c, ratio=0.5, dedup=0.95):
    """强度为 c 的一个单元内的选样顺序。
    边界分：c 与相邻等级之间的 CORAL 阈值（c−1|c 与 c|c+1，两端等级只有一个）上 |σ(z_j) − 0.5| 的最小值，
            越小越靠近强度边界、越难分；代表分：到单元特征均值的距离。
    两个列表按 ratio（边界样本占比）交替合并，再贪心去重：与已选样本最大余弦相似度 > dedup 的片段挪到最后。"""
    p = torch.sigmoid(logits.float())
    js = [j for j in (c - 1, c) if 0 <= j < p.shape[1]]
    margin = (p[:, js] - 0.5).abs().min(1).values
    e = F.normalize(emb.float(), dim=1)
    rep = ((e - e.mean(0)) ** 2).sum(1)
    b_list, r_list = margin.argsort().tolist(), rep.argsort().tolist()
    merged, used, bi, ri, nb = [], set(), 0, 0, 0
    while len(merged) < len(e):
        take_b = nb < ratio * (len(merged) + 1) and bi < len(b_list) or ri >= len(r_list)
        src = b_list if take_b else r_list
        while True:
            i = src[bi] if take_b else src[ri]
            if take_b:
                bi += 1
            else:
                ri += 1
            if i not in used:
                break
        used.add(i)
        merged.append(i)
        nb += take_b
    keep, skipped, sel = [], [], torch.empty(0, e.shape[1], device=e.device)
    for i in merged:
        if len(sel) and float((sel @ e[i]).max()) > dedup:
            skipped.append(i)
            continue
        keep.append(i)
        sel = torch.cat([sel, e[i:i + 1]])
    return keep + skipped


class VideoMemory:
    def __init__(self, codec, budget_mb, select='random', seed=0, num_workers=8, anchor=False, alloc='cell'):
        """alloc：字节配额怎么分。'cell' = 每个 (物种, 强度) 格子平分（默认）；
        'species' = 每个物种平分，物种内不分强度、随机保留；
        'prop' = 每个格子按它在训练集里的片段数成比例分（等价于把所有旧片段放在一起随机保留）。"""
        self.codec, self.budget, self.select, self.alloc = codec, budget_mb * MB, select, alloc
        self.sizes = {}            # 格子 -> 训练集片段数（alloc='prop' 用）
        self.anchor_bytes = 2 * 1024 if anchor else 0
        self.reserved = 0          # 预算里另作他用的字节（如每物种修复网络的权重）
        self.rng = np.random.RandomState(seed)
        self.cells = {}            # (species_idx, c) -> [(payload, nbytes, label, logits, anchor), ...] 按优先级
        self.num_workers = num_workers

    def __len__(self):
        return sum(len(v) for v in self.cells.values())

    def nbytes(self):
        return sum(e[1] for v in self.cells.values() for e in v)

    def _trim(self, quotas):
        for key, entries in self.cells.items():
            quota = quotas[key]
            used, keep = 0, []
            for e in entries:
                if used + e[1] > quota:
                    break
                keep.append(e)
                used += e[1]
            self.cells[key] = keep

    def add_species(self, k, items, emb, logits):
        """items: 物种 k 的训练样本；emb / logits: 模型在这些原始片段上的特征和输出（与 items 对齐，CPU）。"""
        labels = np.array([it['label'] for it in items])
        if self.alloc == 'species':          # 整个物种一个格子，标签 -1；物种内随机
            groups = {(k, -1): np.arange(len(items))}
        else:
            groups = {(k, c): np.nonzero(labels == c)[0] for c in range(C) if (labels == c).any()}
        self.sizes.update({key: len(idx) for key, idx in groups.items()})
        keys = list(self.cells) + list(groups)
        avail = self.budget - self.reserved
        if self.alloc == 'prop':
            total = sum(self.sizes[key] for key in keys)
            quotas = {key: avail * self.sizes[key] // total for key in keys}
        else:
            quotas = {key: avail // len(keys) for key in keys}
        self._trim(quotas)
        for (_, c), idx in groups.items():
            quota = quotas[(k, c)]
            if self.select == 'herding':
                order = idx[herding_order(emb[idx].cuda(), min(len(idx), 2000))]
            elif self.select == 'kcenter':
                order = idx[kcenter_order(emb[idx].cuda(), len(idx))]
            elif self.select == 'boundary':
                order = idx[boundary_order(emb[idx].cuda(), logits[idx].cuda(), c)]
            else:
                order = idx[self.rng.permutation(len(idx))]
            entries, used = [], 0
            # 按优先级分块编码，配额用完即停（变长编码无法事先知道能存几个）
            for s in range(0, len(order), 256):
                chunk = order[s:s + 256]
                loader = DataLoader(_EncodeDataset([items[i] for i in chunk], self.codec), batch_size=None,
                                    num_workers=self.num_workers)
                full = False
                for i, (p, n) in zip(chunk, loader):
                    n += self.anchor_bytes
                    if used + n > quota:
                        full = True
                        break
                    entries.append((p, n, int(labels[i]), logits[i].clone(), emb[i].half().clone()))
                    used += n
                if full:
                    break
            self.cells[(k, c)] = entries

    def loader(self, batch_size):
        entries = [(k, e) for (k, _), v in self.cells.items() for e in v]
        return DataLoader(_ReplayDataset(entries, self.codec), batch_size=batch_size,
                          sampler=RandomSampler(entries, replacement=True, num_samples=10 ** 9),
                          num_workers=self.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=False, prefetch_factor=4)

    def sequential_loader(self, batch_size):
        """按顺序遍历全部记忆样本一次（重拟合分类头时提特征用）。"""
        entries = [(k, e) for (k, _), v in self.cells.items() for e in v]
        return DataLoader(_ReplayDataset(entries, self.codec), batch_size=batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def stats(self):
        n = len(self)
        return {'n_exemplars': n, 'MB': self.nbytes() / MB, 'reserved_MB': self.reserved / MB, 'KB_per_exemplar': self.nbytes() / 1024 / max(n, 1),
                'per_cell': {f'{k}_{c}': len(v) for (k, c), v in sorted(self.cells.items())}}
