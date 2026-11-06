"""视频骨干（默认 S3D；另有 X3D-M、MC3-18，均为 Kinetics-400 预训练）+ 强度头（CORAL 序数头或 4 类 softmax），
Phase 0 适应与端到端增量训练共用。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vcil.models.s3d import S3D, load_s3d_kinetics


def coral_loss(logits, targets, num_classes=4):
    ordinal = torch.stack([(targets > k).float() for k in range(num_classes - 1)], dim=1)
    return F.binary_cross_entropy_with_logits(logits, ordinal)


def coral_predict(logits):
    return torch.sigmoid(logits).sum(dim=-1).round().long().clamp(0, 3)


def make_loss_and_predict(head):
    """head='coral': 与 AquaLytic 一致的序数头；head='ce': 普通 4 类 softmax 交叉熵（对照）。"""
    if head == 'coral':
        return coral_loss, coral_predict
    return F.cross_entropy, lambda logits: logits.argmax(dim=-1)


class TorchvisionVideoBackbone(nn.Module):
    """包装 torchvision / pytorchvideo 的 Kinetics-400 视频模型，接口与 S3D 一致：
    输入 uint8 (B,T,H,W,3) 或 [0,1] float (B,T,3,H,W)，输出 {'embedding': (B, dim)}。"""

    def __init__(self, name):
        super().__init__()
        if name == 'x3d_m':
            from pytorchvideo.models.hub import x3d_m
            net = x3d_m(pretrained=True)
            net.blocks[-1].proj = nn.Identity()          # 去掉 400 类分类层，保留 2048 维池化特征
            net.blocks[-1].activation = None
            mean, std, self.dim = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225), 2048
        elif name == 'mc3_18':
            from torchvision.models.video import MC3_18_Weights, mc3_18
            net = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)
            net.fc = nn.Identity()
            mean, std, self.dim = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989), 512
        else:
            raise ValueError(name)
        self.net = net
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1, 1))

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)
        else:
            x = x.transpose(1, 2)
        emb = self.net((x - self.mean) / self.std).flatten(1)
        return {'embedding': emb}


def build_backbone(name):
    if name == 's3d':
        m = load_s3d_kinetics(S3D(400))
        m.dim = 1024
        return m
    return TorchvisionVideoBackbone(name)


class VideoModel(nn.Module):
    """mlp=False: dropout + 线性头（Phase 0 适应时用的结构）。
    mlp=True : LayerNorm → 512 → ReLU → Dropout → 输出。冻结特征上线性 CORAL 头只有约 52%，
               MLP 头约 86%（7 个物种的特征不是线性可分的），所以端到端增量学习用 MLP 头。"""

    def __init__(self, head='coral', dropout=0.5, mlp=False, backbone='s3d'):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = build_backbone(backbone)
        d = self.backbone.dim
        out = 3 if head == 'coral' else 4                          # CORAL: 3 个阈值；CE: 4 类
        if mlp:
            self.dropout = nn.Identity()
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(512, out))
        else:
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d, out)

    def forward(self, x):
        emb = self.backbone(x)['embedding']
        return self.head(self.dropout(emb)), emb
