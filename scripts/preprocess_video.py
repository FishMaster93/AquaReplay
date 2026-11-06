"""
Convert the released mp4 clips into the arrays used by the code: 16 frames sampled uniformly over the clip,
resized to 224x224 (area interpolation), stored as uint8 arrays of shape (16, 224, 224, 3) in .npy files.
The directory layout is kept, so every clip keeps its session / species / intensity / name.

  Guangzhou (six species, https://zenodo.org/records/23623291):
    python scripts/preprocess_video.py --dataset gz --src <root>/gz/video_dataset --dst <root>/gz/video_npy
  Oplegnathus punctatus (Yantai, https://doi.org/10.5281/zenodo.12759731):
    python scripts/preprocess_video.py --dataset opleg --src <root>/opleg/video_dataset --dst <root>/opleg/video_npy

For O. punctatus the file names "*_video_N.mp4" are renamed to "*_audio_N.npy" so that they match the sample ids
in data/splits/split_clip.csv. Existing outputs are skipped, so the script can be resumed.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import av
import numpy as np

NUM_FRAMES = 16
SIZE = 224


def list_jobs(dataset, src, dst):
    jobs = []
    for dirpath, _, files in sorted(os.walk(src)):
        for fn in sorted(files):
            if not fn.endswith('.mp4'):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), src)[:-4]
            if dataset == 'opleg':
                rel = rel.replace('_video_', '_audio_')
            jobs.append((os.path.join(dirpath, fn), os.path.join(dst, rel + '.npy')))
    return jobs


def decode_clip(path):
    """mp4 -> (16, 224, 224, 3) uint8; all frames are decoded, 16 are sampled uniformly and resized."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        frames = list(container.decode(video=0))
    if not frames:
        raise RuntimeError('no frames decoded')
    idx = np.linspace(0, len(frames) - 1, NUM_FRAMES).round().astype(int)
    out = np.empty((NUM_FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        out[k] = frames[i].reformat(width=SIZE, height=SIZE, format='rgb24', interpolation='AREA').to_ndarray()
    return out


def work(job):
    src, dst = job
    if os.path.exists(dst):
        return None
    try:
        clip = decode_clip(src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + '.tmp.npy'
        np.save(tmp, clip)
        os.replace(tmp, dst)
        return None
    except Exception as e:
        return f'{src}: {e!r}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['gz', 'opleg'], required=True)
    p.add_argument('--src', required=True, help='directory with the mp4 clips')
    p.add_argument('--dst', required=True, help='output directory for the .npy arrays')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    jobs = list_jobs(a.dataset, a.src, a.dst)
    print(f'{len(jobs)} clips found')
    errors = []
    with ProcessPoolExecutor(a.workers) as ex:
        for n, err in enumerate(ex.map(work, jobs, chunksize=16), 1):
            if err:
                errors.append(err)
            if n % 2000 == 0:
                print(f'{n}/{len(jobs)} done', flush=True)
    print(f'finished, {len(errors)} errors')
    for e in errors[:20]:
        print(' ', e)


if __name__ == '__main__':
    main()
