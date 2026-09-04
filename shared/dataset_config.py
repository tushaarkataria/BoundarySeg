"""
Central dataset configuration for all semi-supervised segmentation baselines.

Usage in any training script:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', 'shared'))   # adjust depth as needed
    from dataset_config import configure_from_dataset

    parser.add_argument('--dataset', default='LA',
                        choices=['LA', 'VERSE', 'ACDC'])
    args = parser.parse_args()
    configure_from_dataset(args)
    # args now has: root_path, num_classes, patch_size, max_samples
"""

import os as _os

# Data and results roots.
#
# Point BOUNDARYSEG_DATA_ROOT at the directory holding the prepared datasets
# (one subdirectory per dataset: LA/, VERSE/, ...) and BOUNDARYSEG_RESULTS_ROOT
# at where runs should be written. Both default to ./data and ./results beside
# this checkout, so a fresh clone works with no environment set up as long as
# the data is symlinked in.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

DATA_ROOT    = _os.environ.get('BOUNDARYSEG_DATA_ROOT',
                               _os.path.join(_REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('BOUNDARYSEG_RESULTS_ROOT',
                               _os.path.join(_REPO_ROOT, 'results'))

# One entry per dataset.
# root        : absolute path passed to the dataloader as base_dir / root
# num_classes : total classes including background (2 = binary)
# patch_size  : (D, H, W) training patch
# max_samples : total training cases (labeled + unlabeled)
_CFG = {
    'LA': {
        'root':        f'{DATA_ROOT}/LA',
        'num_classes': 2,
        'patch_size':  (112, 112, 80),
        'max_samples': 79,
    },
    'VERSE': {
        'root':        f'{DATA_ROOT}/VERSE',
        'num_classes': 2,
        'patch_size':  (128, 128, 128),
        'max_samples': 190,
    },
    'ACDC': {
        # nnU-Net / UNETR++ raw-data layout (dataset.json + imagesTr/labelsTr).
        'root':        f'{DATA_ROOT}/ACDC',
        'num_classes': 4,
        # Native volumes are anisotropic, ~10 slices deep. Z=16 is the
        # minimum that survives VNet's 4 stride-2 downsamples (16->8->4->2->1);
        # anything smaller fails with "kernel size can't be greater than
        # input size" on the last downsampling block.
        'patch_size':  (160, 160, 16),
        'max_samples': 160,
    },
}


def get_dataset_config(name):
    if name not in _CFG:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(_CFG)}")
    return dict(_CFG[name])


def configure_from_dataset(args):
    """
    Fill in args.root_path, args.num_classes, args.patch_size, args.max_samples
    from args.dataset.  Call this immediately after parser.parse_args().

    Also sets args.num_fg = num_classes - 1 (foreground channels for the model).
    """
    cfg = get_dataset_config(args.dataset)
    args.root_path          = cfg['root']
    args.num_classes        = cfg['num_classes']
    args.patch_size         = cfg['patch_size']
    args.dataset_max_samples = cfg['max_samples']  # natural training set size
    args.num_fg             = cfg['num_classes'] - 1
    args.results_root       = RESULTS_ROOT
    if not getattr(args, 'save_root', None):
        args.save_root = RESULTS_ROOT
    # Only set max_samples if the script didn't receive an explicit override.
    # Scripts should use default=None for --max_samples so None signals
    # "not explicitly set by the user".
    if getattr(args, 'max_samples', None) is None:
        args.max_samples = cfg['max_samples']
    return args
