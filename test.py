#!/usr/bin/env python
"""
Evaluate a trained BoundarySEG checkpoint on a dataset's held-out test set.

Training already evaluates its own final checkpoint (see `eval_last/` beside
each run). This script is the standalone path: point it at any `.pth` and it
reproduces exactly the same numbers, so the released checkpoints can be scored
without retraining, and so a competing method can be scored by the identical
protocol.

  # score one released checkpoint
  python test.py --dataset LA --checkpoint checkpoints/LA_4lbl_4unl/seed1337.pth

  # score all three seeds of a split and print mean +/- std
  bash scripts/test_checkpoints.sh LA_4lbl_4unl

Protocol (identical to training's own final evaluation):
  * sliding-window inference at the dataset's training patch size
  * argmax over [implicit background, foreground logits]
  * binary datasets keep only the largest connected component
  * Dice / Jaccard / HD95 / ASD via medpy, in voxel units

Results are written to <out>/eval_results.csv (one row per case plus a MEAN
row) and printed to the terminal.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.join(_HERE, 'shared'))

from networks.vnet_sdf import VNet_mine                                # noqa: E402
from dataset_config import configure_from_dataset, get_dataset_config  # noqa: E402
from evaluation import evaluate_and_save, build_test_pairs             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True,
                    help='Path to a .pth state dict (e.g. a run\'s last_model.pth).')
    ap.add_argument('--dataset', default='LA', choices=['LA', 'VERSE', 'ACDC'])
    ap.add_argument('--data_root', default=None,
                    help='Override the dataset directory. Defaults to '
                         '<repo>/data/<DATASET> (see README: symlink it there).')
    ap.add_argument('--out', default=None,
                    help='Directory for eval_results.csv and predictions '
                         '(default: <checkpoint dir>/eval_<checkpoint name>).')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--normalization', default=None,
                    choices=['batchnorm', 'instancenorm', 'groupnorm'],
                    help='Must match the checkpoint. Default: instancenorm for '
                         'LA/VERSE, batchnorm for ACDC -- the paper settings.')
    ap.add_argument('--max_cases', type=int, default=0,
                    help='If >0, evaluate only the first N test cases (smoke runs).')
    ap.add_argument('--save_predictions', action='store_true',
                    help='Keep the predicted masks (NIfTI) written during '
                         'evaluation. Off by default -- they are ~1 GB per run.')
    ap.add_argument('--labelnum', type=int, default=0,
                    help='Recorded in the CSV for bookkeeping only; does not '
                         'affect any metric.')
    ap.add_argument('--unlabelnum', type=int, default=0,
                    help='Recorded in the CSV for bookkeeping only; does not '
                         'affect any metric.')
    args = ap.parse_args()

    # Honor an externally-set CUDA_VISIBLE_DEVICES. Assigning it unconditionally
    # would silently overwrite the caller's choice -- `CUDA_VISIBLE_DEVICES=1
    # python train.py` would land on GPU 0, because --gpu defaults to '0' and
    # args.gpu is the only device selector in this file (everything else just
    # calls .cuda(), i.e. device 0 of whatever is visible).
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # ACDC's champion uses BatchNorm: InstanceNorm destabilises 2 of 3 seeds
    # there. LA and VerSe use InstanceNorm. Loading with the wrong one silently
    # produces garbage rather than an error, so it is pinned by dataset.
    if args.normalization is None:
        args.normalization = 'batchnorm' if args.dataset == 'ACDC' else 'instancenorm'

    cfg = get_dataset_config(args.dataset)
    configure_from_dataset(args)
    if args.data_root:
        args.root_path = args.data_root
    num_fg = cfg['num_classes'] - 1

    net = VNet_mine(n_channels=1, n_classes=num_fg,
                    normalization=args.normalization, has_dropout=False).cuda()
    state = torch.load(args.checkpoint, map_location='cuda')
    net.load_state_dict(state)
    net.eval()

    pairs = build_test_pairs(args)
    if args.max_cases > 0:
        pairs = pairs[:args.max_cases]

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        'eval_' + os.path.splitext(os.path.basename(args.checkpoint))[0])

    # The metadata columns the CSV carries are descriptive only; they do not
    # affect any metric.
    args.method = 'surface_ema'
    args.exp = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    args.model = 'vnet'
    # Recorded so that summarize.py can tell three seed CSVs apart. Taken from
    # the filename (seed1337.pth -> 1337); inference itself is deterministic.
    _stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    args.seed = _stem[4:] if _stem.startswith('seed') else _stem
    args.max_iterations = 20000
    # configure_from_dataset() fills max_samples with the FULL training-set size;
    # for a checkpoint that means nothing, so pin it to what was actually used
    # (the CSV derives unlabelnum = max_samples - labelnum from it).
    args.max_samples = args.labelnum + args.unlabelnum
    args.save_root = out          # keep all_results.csv beside the run
    args.tta = False

    print(f'checkpoint : {args.checkpoint}')
    print(f'dataset    : {args.dataset}  ({len(pairs)} test cases)  '
          f'root={args.root_path}')
    print(f'patch      : {tuple(args.patch_size)}   norm={args.normalization}\n')

    evaluate_and_save(net, pairs, args, out, baseline_name='BoundarySEG',
                      experiment=None, iteration=args.max_iterations)

    if not args.save_predictions:
        import shutil
        shutil.rmtree(os.path.join(out, 'test_predictions'), ignore_errors=True)


if __name__ == '__main__':
    main()
