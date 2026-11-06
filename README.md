# AquaReplay

Code for **AquaReplay**, a video-only, exemplar-based method for species-incremental fish feeding intensity
recognition (FFIR). A model recognises four feeding intensity levels (None, Weak, Medium, Strong) and learns new fish
species one at a time without forgetting the earlier ones.

AquaReplay has three components:

1. **Compressed video memory**: stored clips are encoded with H.264 (CRF 33), about 17 KB per 16-frame clip, so a
   200 MB budget holds about 12,000 complete clips with their motion.
2. **Species-specific task-aware restorers**: a small CNN per species (0.25 MB, counted in the budget) restores
   decoded clips, trained with an L1 loss plus a feature loss that keeps the restored clip close to the original in
   the encoder's feature space.
3. **Phase-end head refit**: after each phase the ordinal head is refitted on fixed features of the current data and
   the restored memory.

The backbone is S3D (Kinetics-400 pretrained) with an MLP ordinal head (CORAL-style, three thresholds).

## Repository structure

```
vcil/
  paths.py            paths (all relative to the repository root)
  data.py             species/phase definition, manifest reader, clip dataset, GPU augmentation
  models/s3d.py       S3D backbone
  models/video.py     backbone (S3D / X3D-M / MC3-18) + ordinal head
  e2e/codecs.py       memory formats: raw, single frame, frame subsampling, JPEG, H.264
  e2e/memory.py       byte-budgeted video memory (per species-intensity cell quotas)
  e2e/restore.py      restorer network and its training loop
  e2e/learner.py      incremental learner (AquaReplay and baselines)
scripts/
  preprocess_video.py mp4 -> (16, 224, 224, 3) uint8 .npy arrays
  build_manifest.py   build data/manifest_video.csv from the released split and your local arrays
  train_phase0.py     phase 0: adapt the encoder on red tilapia
  make_mlp_ckpt.py    phase 0: fit the MLP ordinal head on frozen features
  train_restorer.py   pretrain the first restorer R0
  run_e2e.py          run one incremental experiment (any method)
  run_paper.sh        reproduce the main comparison
  make_*_fig.py, make_figures.py, measure_latency.py   figures and latency of the paper
data/splits/split_clip.csv   clip-level 80/10/10 split (99,960 clips), stratified by species and intensity
```

## Installation

```bash
conda create -n aquareplay python=3.11 -y
conda activate aquareplay
pip install -r requirements.txt
```

PyAV is used for H.264 encoding and decoding of the memory; a CUDA GPU is required for training.

## Data

Seven species are learned in seven phases:

| Phase | Species | Site |
|---|---|---|
| 0 | Red tilapia (`Red_tilapia_2` + `Red_tilapia_3`) | Guangzhou |
| 1 | Tilapia | Guangzhou |
| 2 | Jade perch | Guangzhou |
| 3 | Largemouth bass | Guangzhou |
| 4 | Lotus carp | Guangzhou |
| 5 | Sunfish | Guangzhou |
| 6 | *Oplegnathus punctatus* | Yantai |

1. **Download** the Guangzhou clips (six species) from Zenodo: https://zenodo.org/records/23623291
   (`video_dataset.tar`, 28 GB), and the *O. punctatus* clips from https://doi.org/10.5281/zenodo.12759731.

   ```bash
   mkdir -p <data_root>/gz && tar -xf video_dataset.tar -C <data_root>/gz/
   # -> <data_root>/gz/video_dataset/<session>/<Species>/<Intensity>/<n>.mp4
   ```

2. **Preprocess** every clip into 16 uniformly sampled frames at 224 × 224:

   ```bash
   python scripts/preprocess_video.py --dataset gz    --src <data_root>/gz/video_dataset    --dst <data_root>/gz/video_npy
   python scripts/preprocess_video.py --dataset opleg --src <data_root>/opleg/video_dataset --dst <data_root>/opleg/video_npy
   ```

3. **Build the manifest** with the released split:

   ```bash
   python scripts/build_manifest.py --gz_root <data_root>/gz/video_npy --opleg_root <data_root>/opleg/video_npy
   # -> data/manifest_video.csv (99,960 clips)
   ```

The Zenodo clips are transcoded to 854 × 480. The paper used the original-resolution recordings, so numbers
reproduced from the released clips may differ slightly.

## Pretrained backbone

Download the Kinetics-400 S3D weights `S3D_kinetics400.pt` from https://github.com/kylemin/S3D and save them as
`pretrained/S3D400.pt`. The X3D-M and MC3-18 weights are downloaded automatically when those backbones are used.

## Running the code

All commands are run from the repository root. Checkpoints go to `workspace/`, results to
`results/e2e/clip/<method>_b<budget>_<tag>_seed<seed>.json` and logs to `logs/e2e/`.

**Step 1. Phase 0** (encoder adaptation on red tilapia, then the MLP head on frozen features):

```bash
python scripts/train_phase0.py                  # -> workspace/S3D_phase0_clip/phase0_best.pt
python scripts/make_mlp_ckpt.py                 # -> workspace/S3D_phase0_clip/phase0_mlp.pt
```

**Step 2. First restorer R0** (pixel loss over CRF 18/23/28/33, then fine-tuned with the feature loss):

```bash
python scripts/train_restorer.py --steps 15000 --bs 4 --out workspace/restorer/restorer.pt
python scripts/train_restorer.py --feat_w 0.2 --init workspace/restorer/restorer.pt --steps 6000 --bs 2 \
    --out workspace/restorer/restorer_task0.2.pt
```

**Step 3. AquaReplay** (phases 1–6, 200 MB budget):

```bash
python scripts/run_e2e.py --method aquareplay \
    --ckpt workspace/S3D_phase0_clip/phase0_mlp.pt \
    --restore workspace/restorer/restorer_task0.2.pt \
    --budget_mb 200 --seed 42 --tag main
```

At each phase the encoder and head are trained for 3 epochs on the new species mixed 1:1 with restored memory clips,
a restorer is trained for the new species, the memory is updated within the budget, and the head is refitted.

**Step 4. Baselines and the upper bound.** `scripts/run_paper.sh` runs steps 1–3 and all baselines
(ER, DER++, SMILE, vCLIMB, MRDC, iCaRL, fine-tuning, LwF, EWC, joint training):

```bash
bash scripts/run_paper.sh                       # seed 42
SEEDS="42 43 44" bash scripts/run_paper.sh      # three seeds
```

### Other experiments

Add the options below to the AquaReplay command in step 3 (use a different `--tag` for each):

| Experiment | Options |
|---|---|
| Memory budget | `--budget_mb 20` / `50` / `500` / `1000` |
| Compression rate | `--codec h264crf28` / `h264crf38` |
| No restorer | `--method er_h264 --refit_epochs 30 --refit_balance 0` (without `--restore`) |
| One shared restorer | `--restore_per_species 0` |
| Pixel-only restorer | `--restore workspace/restorer/restorer.pt --restore_w 0` |
| No head refit | `--refit_epochs 0` |
| Frozen encoder | `--epochs 0` |
| Species order | `--order 6 5 4 3 2 1` (phases 1–6; red tilapia is always phase 0) |
| Memory allocation | `--alloc species` (equal per species) / `--alloc prop` (proportional to data) |

**Other backbones** (X3D-M or MC3-18): run steps 1–2 with the backbone and its own checkpoints, then step 3:

```bash
python scripts/train_phase0.py --backbone x3d_m --exp_name X3D_phase0_clip
python scripts/make_mlp_ckpt.py --src workspace/X3D_phase0_clip/phase0_best.pt --out workspace/X3D_phase0_clip/phase0_mlp.pt
python scripts/train_restorer.py --phase0 workspace/X3D_phase0_clip/phase0_best.pt --feat_w 0.2 \
    --init workspace/restorer/restorer.pt --steps 6000 --bs 2 --out workspace/restorer/restorer_x3d.pt
python scripts/run_e2e.py --method aquareplay --ckpt workspace/X3D_phase0_clip/phase0_mlp.pt \
    --restore workspace/restorer/restorer_x3d.pt --tag x3d
```

## Metrics

After every phase the model is evaluated on the test sets of all species learned so far. Each result file reports:

- `final_acc`: accuracy on all seven species after the last phase;
- `avg_inc_acc`: mean of the accuracies after each phase, including phase 0;
- `final_avg_forget`: mean drop of each earlier species from its best accuracy to its final accuracy;
- `history`: per-phase accuracy on each species, and `memory`: number of stored clips and their size.

## Figures

After the experiments have finished (`--save_ckpt 1` is needed for the restorer figure):

```bash
python scripts/make_phase_fig.py       # accuracy and forgetting across phases
python scripts/make_budget_fig.py      # accuracy vs. memory budget and number of stored clips
python scripts/make_figures.py         # per-species accuracy after each phase
python scripts/make_restore_fig.py     # restorer statistics over the test set (also caches them)
python scripts/make_restore_fig2.py    # restorer examples from the two sites
python scripts/measure_latency.py      # inference latency of the backbones
```

Figures are written to `paper_figures/`.
