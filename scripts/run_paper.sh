#!/bin/bash
# Reproduce the main experiments of the paper (200 MB budget, seed 42 unless changed by SEEDS).
# Run from the repository root after preparing data/manifest_video.csv and pretrained/S3D400.pt (see README).
set -e
PY=${PY:-python}
SEEDS=${SEEDS:-42}
CK=workspace/S3D_phase0_clip/phase0_mlp.pt
R0=workspace/restorer/restorer_task0.2.pt
mkdir -p logs/e2e

# 1. Phase 0: adapt S3D on red tilapia, then fit the MLP ordinal head on frozen features
$PY scripts/train_phase0.py
$PY scripts/make_mlp_ckpt.py

# 2. Pretrain the first restorer R0 (pixel loss on several CRFs, then + 0.2 x feature loss)
$PY scripts/train_restorer.py --steps 15000 --bs 4 --out workspace/restorer/restorer.pt
$PY scripts/train_restorer.py --feat_w 0.2 --init workspace/restorer/restorer.pt --steps 6000 --bs 2 --out $R0

B="--ckpt $CK --refit_epochs 30 --refit_balance 0"
for s in $SEEDS; do
  # 3. AquaReplay
  $PY scripts/run_e2e.py --method aquareplay --ckpt $CK --restore $R0 --seed $s --tag main --save_ckpt 1
  # 4. Exemplar-based baselines (same budget and stage-end head refit)
  for m in er smile vclimb mrdc; do $PY scripts/run_e2e.py --method $m $B --seed $s --tag main; done
  $PY scripts/run_e2e.py --method der $B --der_alpha 0.02 --seed $s --tag main
  $PY scripts/run_e2e.py --method icarl --ckpt $CK --seed $s --tag main
  # 5. Exemplar-free baselines and the joint-training upper bound
  $PY scripts/run_e2e.py --method finetune --ckpt $CK --refit_epochs 30 --seed $s --tag main
  $PY scripts/run_e2e.py --method finetune --ckpt $CK --kd_out 1 --seed $s --tag lwf
  $PY scripts/run_e2e.py --method finetune --ckpt $CK --ewc 1000 --seed $s --tag ewc
  $PY scripts/run_e2e.py --method joint $B --seed $s --tag main
done
