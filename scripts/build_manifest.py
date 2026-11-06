"""
Build data/manifest_video.csv from the released split (data/splits/split_clip.csv) and the local .npy arrays
produced by scripts/preprocess_video.py.

  python scripts/build_manifest.py --gz_root <root>/gz/video_npy --opleg_root <root>/opleg/video_npy

A sample id has the form  gz/<session>/<Species>/<Intensity>/<n>  or  opleg/<date>/<feeding>/<intensity>/<name>,
and its array is expected at  <gz_root or opleg_root>/<rest of the id>.npy.
The split is clip-level 80/10/10, stratified by species and intensity (99,960 clips).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.paths import MANIFEST, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gz_root', required=True, help='.npy root of the six Guangzhou species')
    p.add_argument('--opleg_root', required=True, help='.npy root of Oplegnathus punctatus')
    p.add_argument('--split', default=str(ROOT / 'data' / 'splits' / 'split_clip.csv'))
    p.add_argument('--out', default=str(MANIFEST))
    a = p.parse_args()
    roots = {'gz': a.gz_root, 'opleg': a.opleg_root}
    rows, missing = [], []
    for r in csv.DictReader(open(a.split, newline='')):
        site, rel = r['sample_id'].split('/', 1)
        path = os.path.join(roots[site], rel + '.npy')
        if not os.path.exists(path):
            missing.append(path)
            continue
        rows.append({'sample_id': r['sample_id'], 'site': site, 'species': r['species'], 'session': r['session'],
                     'intensity': r['intensity'], 'label': r['label'], 'video_path': path, 'split_clip': r['split']})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} clips -> {a.out}; {len(missing)} missing')
    for m in missing[:10]:
        print('  missing', m)


if __name__ == '__main__':
    main()
