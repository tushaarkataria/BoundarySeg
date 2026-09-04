#!/usr/bin/env python
"""
Aggregate per-case evaluation CSVs into the mean +/- std table the paper reports.

Every evaluation writes an `eval_results.csv` holding one row per test case plus
a final `MEAN` row. This reads the MEAN row of each CSV given, treats each as one
seed, and reports mean +/- std ACROSS SEEDS -- which is what the paper's error
bars are (not the across-case spread).

  python tools/summarize.py results/LA/**/eval_last/eval_results.csv
  python tools/summarize.py --label "LA 4:4" checkpoints/LA_4lbl_4unl/eval_*/eval_results.csv

Dice is printed as a percentage; HD95 and ASD in voxels, as in the paper.
"""

import argparse
import csv
import os
import statistics as st
import sys


def _mean_row(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get('case_id') == 'MEAN':
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csvs', nargs='+', help='One eval_results.csv per seed.')
    ap.add_argument('--label', default=None, help='Row label for the printed table.')
    args = ap.parse_args()

    # A run that is still training simply has no CSV yet; say so rather than
    # silently averaging over fewer seeds than the caller thinks.
    missing = [p for p in args.csvs if not os.path.exists(p)]
    for p in missing:
        print(f'  missing: {p}', file=sys.stderr)
    found = [p for p in args.csvs if p not in missing]
    if not found:
        print('No eval_results.csv found. Has training finished?', file=sys.stderr)
        return 1

    per_seed = {}
    multiclass = False
    for p in found:
        r = _mean_row(p)
        if r is None:
            print(f'  no MEAN row: {p}', file=sys.stderr)
            continue
        seed = r.get('seed') or os.path.basename(os.path.dirname(p))
        if r.get('mean_dice'):
            multiclass = True
            per_seed[seed] = (float(r['mean_dice']), float(r['mean_hd95']), None)
        else:
            per_seed[seed] = (float(r['dice']), float(r['hd95']), float(r['asd']))

    if not per_seed:
        print('No usable rows.', file=sys.stderr)
        return 1

    def agg(vals, scale=1.0):
        vals = [v * scale for v in vals]
        s = st.stdev(vals) if len(vals) > 1 else 0.0
        return f'{st.mean(vals):.2f}+/-{s:.2f}'

    dices = [v[0] for v in per_seed.values()]
    # CSVs store Dice in [0,1]; the paper reports percent.
    scale = 100.0 if max(dices) <= 1.5 else 1.0

    label = args.label or 'result'
    print()
    print(f'{"":<22} {"Dice":>16} {"HD95":>16} {"ASD":>16}')
    print('-' * 74)
    asd = '-' if multiclass else agg([v[2] for v in per_seed.values()])
    print(f'{label:<22} {agg(dices, scale):>16} '
          f'{agg([v[1] for v in per_seed.values()]):>16} {asd:>16}')
    print(f'\n{len(per_seed)} seed(s): {", ".join(sorted(per_seed))}')
    for seed in sorted(per_seed):
        d, h, a = per_seed[seed]
        a = '-' if a is None else f'{a:.2f}'
        print(f'  seed {seed:<8} Dice {d * scale:6.2f}   HD95 {h:6.2f}   ASD {a:>6}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
