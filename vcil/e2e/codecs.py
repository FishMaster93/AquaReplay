"""
回放记忆的样本存储格式（编码器）。每种有样本方法对应一种格式，记忆按实际字节数计预算。
编解码都在 CPU 上做（放在 DataLoader worker 里并行），接口：
  encode(clip) -> (payload, nbytes)   clip: (16,224,224,3) uint8 CPU
  decode(payload) -> (16,224,224,3) uint8 CPU

  raw     : 原始 16 帧（ER / DER++）
  frame   : 只存中间 1 帧，回放时重复成 16 帧（SMILE）
  subsamp : 均匀存 k 帧原始帧，回放时最近邻重复回 16 帧（vCLIMB）
  jpeg    : 16 帧逐帧 JPEG（MRDC）
  h264    : 16 帧 H.264 编码
  rawref  : 上限实验用：回放原始片段（从磁盘按路径读），但按给定的 KB/段 计预算，
            样本数与对应 H.264 档位相同，用来衡量"压缩再还原"的损伤上限
  motion  : 外观-运动分离 v1：1 张低分辨率 JPEG 关键帧 + 活动区域块的量化残差（zlib）。
            同等字节下明显不如 H.264（98 KB/段 60.3% vs H.264 72 KB/段 83.3%），保留作消融记录
"""
import io
import zlib

import av
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, encode_jpeg

T, H = 16, 224


def _repeat_to_T(f, k):
    """(k,...) -> (T,...)，最近邻重复"""
    return f[torch.div(torch.arange(T) * k, T, rounding_mode='floor')]


class Raw:
    name = 'raw'

    def encode(self, clip):
        return clip.clone(), clip.numel()

    def decode(self, p):
        return p


class RawRef:
    by_ref = True

    def __init__(self, kb):
        self.kb = kb
        self.name = f'rawref{kb:g}'

    def encode_ref(self, path):
        return path, int(self.kb * 1024)

    def decode(self, p):
        return torch.from_numpy(np.load(p))


class Subsample:
    """k=1 即 SMILE（单帧，取中间帧），k>1 为 vCLIMB 式均匀少帧存储。"""

    def __init__(self, k):
        self.k = k
        self.name = 'frame' if k == 1 else f'subsamp{k}'
        self.idx = torch.tensor([T // 2]) if k == 1 else torch.linspace(0, T - 1, k).round().long()

    def encode(self, clip):
        f = clip[self.idx].clone()
        return f, f.numel()

    def decode(self, p):
        return _repeat_to_T(p, self.k)


class JPEG:
    def __init__(self, quality):
        self.q = quality
        self.name = f'jpeg{quality}'

    def encode(self, clip):
        enc = encode_jpeg(list(clip.permute(0, 3, 1, 2).contiguous()), quality=self.q)
        return enc, sum(e.numel() for e in enc)

    def decode(self, p):
        return torch.stack(decode_jpeg(p)).permute(0, 2, 3, 1).contiguous()


class H264:
    def __init__(self, crf):
        self.crf = crf
        self.name = f'h264crf{crf}'

    def encode(self, clip):
        buf = io.BytesIO()
        with av.open(buf, 'w', format='mp4') as out:
            s = out.add_stream('libx264', rate=8)
            s.width = s.height = H
            s.pix_fmt = 'yuv420p'
            s.options = {'crf': str(self.crf), 'preset': 'medium'}
            s.thread_count = 1
            for fr in clip.numpy():
                for pkt in s.encode(av.VideoFrame.from_ndarray(fr, format='rgb24')):
                    out.mux(pkt)
            for pkt in s.encode():
                out.mux(pkt)
        data = buf.getvalue()
        return data, len(data)

    def decode(self, p):
        with av.open(io.BytesIO(p)) as inp:
            frames = [fr.to_ndarray(format='rgb24') for fr in inp.decode(video=0)]
        return torch.from_numpy(np.stack(frames[:T]))


class Motion:
    """外观-运动分离 v1：
      k : 中间帧缩到 res x res 后 JPEG（质量 key_q），只存一次
      M : 残差 R_t = f_t - k 在 block x block 小块上的平均能量，取前 rho 比例的小块（全片段共用一个掩码）
      R : 掩码内 16 帧残差，按 step 量化成 int8，zlib 无损压缩
    重建 x̂_t = k + M ⊙ R_t，再放大回 224。"""

    def __init__(self, res=112, rho=0.25, block=8, step=4, key_q=85):
        self.res, self.rho, self.block, self.step, self.key_q = res, rho, block, step, key_q
        self.nb = res // block
        self.name = f'motion_r{res}_p{int(rho * 100)}_s{step}'

    def encode(self, clip):
        b, nb, r = self.block, self.nb, self.res
        f = F.interpolate(clip.permute(0, 3, 1, 2).float(), size=(r, r), mode='area')       # (16,3,r,r)
        key_bytes = encode_jpeg(f[T // 2].round().clamp(0, 255).to(torch.uint8), quality=self.key_q)
        key = decode_jpeg(key_bytes).float()
        q = torch.round((f - key[None]) / self.step).clamp(-127, 127)
        energy = F.avg_pool2d(q.abs().mean(dim=(0, 1))[None, None], b)[0, 0].flatten()
        n_keep = max(1, int(round(self.rho * nb * nb)))
        mask = torch.zeros(nb * nb, dtype=torch.bool)
        mask[energy.topk(n_keep).indices] = True
        mask = mask.view(nb, nb)
        blocks = q.to(torch.int8).reshape(T, 3, nb, b, nb, b).permute(2, 4, 0, 1, 3, 5)[mask]   # (K,16,3,b,b)
        res = zlib.compress(blocks.contiguous().numpy().tobytes(), 6)
        bits = np.packbits(mask.numpy().ravel())
        return (key_bytes, bits, res), key_bytes.numel() + bits.nbytes + len(res)

    def decode(self, p):
        key_bytes, bits, res = p
        b, nb, r = self.block, self.nb, self.res
        key = decode_jpeg(key_bytes).float()
        mask = torch.from_numpy(np.unpackbits(bits)[:nb * nb].astype(bool)).view(nb, nb)
        blocks = torch.from_numpy(np.frombuffer(zlib.decompress(res), dtype=np.int8).copy()).view(-1, T, 3, b, b)
        q = torch.zeros(nb, nb, T, 3, b, b, dtype=torch.int8)
        q[mask] = blocks
        q = q.permute(2, 3, 0, 4, 1, 5).reshape(T, 3, r, r)
        rec = (key[None] + q.float() * self.step).clamp(0, 255)
        rec = F.interpolate(rec, size=(H, H), mode='bilinear', align_corners=False)
        return rec.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def build_codec(spec):
    """'raw' / 'frame' / 'subsamp4' / 'jpeg50' / 'h264crf33' / 'motion' / 'motion:res=112,rho=0.25,step=4'"""
    if spec == 'raw':
        return Raw()
    if spec.startswith('rawref'):
        return RawRef(float(spec[len('rawref'):]))
    if spec == 'frame':
        return Subsample(1)
    if spec.startswith('subsamp'):
        return Subsample(int(spec[len('subsamp'):]))
    if spec.startswith('jpeg'):
        return JPEG(int(spec[len('jpeg'):]))
    if spec.startswith('h264crf'):
        return H264(int(spec[len('h264crf'):]))
    if spec.startswith('motion'):
        kw = {}
        if ':' in spec:
            for kv in spec.split(':', 1)[1].split(','):
                k, v = kv.split('=')
                kw[k] = float(v) if k == 'rho' else int(v)
        return Motion(**kw)
    raise ValueError(spec)
