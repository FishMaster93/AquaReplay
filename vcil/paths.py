"""All paths used by the code, relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data' / 'manifest_video.csv'
PRETRAINED = ROOT / 'pretrained'
WORKSPACE = ROOT / 'workspace'      # 编码器训练产物（checkpoint、日志）
FEAT_ROOT = ROOT / 'features'       # 冻结编码器特征缓存
RESULTS = ROOT / 'results'
