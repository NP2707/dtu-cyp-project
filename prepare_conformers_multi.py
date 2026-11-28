"""
Output: <out>/<key>.npz with
    z    [n_atoms]            atomic numbers, shared by every conformer
    pos  [k, n_atoms, 3]      k generated conformers, k <= --num-conformers
    k    scalar               how many were actually generated

Usage:
    python prepare_conformers_multi.py --num-conformers 10 --workers 8
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from atom_jepa.conformers import smiles_to_samples

TASKS = ('regression', 'classification')
SPLITS = ('train', 'val', 'test')


def conformers(job):
    """Generate one molecule's conformers. Returns (key, n_generated, error)."""
    key, smiles, folder, seed, k, mmff_max_iters = job
    path = Path(folder) / f'{key}.npz'
    if path.exists():
        return key, -1, ''
    try:
        samples = smiles_to_samples(smiles, num_conformers=k, seed=seed,
                                    mmff_max_iters=mmff_max_iters)
        if not samples:
            return key, 0, 'nix nada'
        z = samples[0]['atomic_numbers'].numpy()
        positions = []
        for sample in samples:
            other = sample['atomic_numbers'].numpy()
            if other.shape != z.shape or not np.array_equal(other, z):
                return key, 0, 'atom order differs between conformers'
            coordinates = sample['node_coordinates'].numpy()
            if coordinates.shape != (len(z), 3) or not np.isfinite(coordinates).all():
                return key, 0, 'bad coordinate array'
            positions.append(coordinates)
        stacked = np.stack(positions).astype(np.float32)
        temp = path.with_suffix('.partial')
        with temp.open('wb') as handle:
            np.savez_compressed(handle, z=z, pos=stacked, k=np.int64(len(positions)))
        temp.replace(path)
        return key, len(positions), ''
    except Exception as exc:
        return key, 0, str(exc)


def molecule_list(data):
    """Every unique (key, SMILES) across both tasks and all three splits."""
    frames = []
    for task in TASKS:
        for split in SPLITS:
            path = data / task / f'{split}.csv'
            if not path.exists():
                raise SystemExit(f'Missing {path}; run prepare_data.py first')
            frames.append(pd.read_csv(path, usecols=['key', 'SMILES']))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates('key')
    clashing = frame.groupby('key').SMILES.nunique()
    if (clashing > 1).any():
        raise SystemExit('The same key maps to different SMILES; splits are inconsistent')
    return frame.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data'),
                   help='Folder written by prepare_data.py')
    p.add_argument('--num-conformers', type=int, default=10)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mmff-max-iters', type=int, default=500)
    args = p.parse_args()
    if args.num_conformers < 1 or args.workers < 1 or not 0 <= args.seed < 2 ** 31 - 1:
        p.error('num-conformers >= 1, workers >= 1 and a valid seed')

    folder = args.out or args.data / 'conformers_multi'
    folder.mkdir(parents=True, exist_ok=True)
    frame = molecule_list(args.data)
    if args.limit:
        frame = frame.head(args.limit)
    manifest_path = folder / 'manifest.json'
    settings = {'num_conformers': args.num_conformers, 'seed': args.seed,
                'mmff_max_iters': args.mmff_max_iters}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())['settings']
        if previous != settings:
            raise SystemExit(f'{folder} was built with {previous}; use a new --out folder')


    work = [(r.key, r.SMILES, str(folder), args.seed, args.num_conformers,
             args.mmff_max_iters) for r in frame.itertuples()]
    
    
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(conformers, work), 1):
            results.append(result)
            if i % 100 == 0:
                print(f'Conformers {i}/{len(work)}')



    report = pd.DataFrame(results, columns=['key', 'n_conformers', 'error'])
    report.to_csv(folder / 'conformer_multi_report.csv', index=False)
    skipped = report[report.n_conformers < 0]
    failed = report[(report.n_conformers == 0) & (report.error != '')]
    made = report[report.n_conformers > 0]
    counts = made.n_conformers.value_counts().sort_index().to_dict()
    manifest = {'settings': settings, 'molecules': len(report),
                'already_present': len(skipped), 'generated': len(made),
                'failed': len(failed),
                'conformers_per_molecule': {int(k): int(v) for k, v in counts.items()}}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if len(failed):
        print(f'\n{len(failed)} molecules failed; first few:')
        print(failed.head().to_string(index=False))


if __name__ == '__main__':
    main()