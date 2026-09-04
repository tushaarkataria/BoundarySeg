#!/usr/bin/env python
"""
BoundarySEG — Unified training script.
Supports binary (LA, VERSE) and multi-class (ACDC).

═══════════════════════════════════════════════════════════════════════════════
PHASED TRAINING PLAN
═══════════════════════════════════════════════════════════════════════════════

The experiments are structured in 7 phases to tell a clean ablation story.
Each phase isolates one contribution and builds on the previous.
Run all three seeds (1337, 42, 123) per configuration.

──────────────────────────────────────────────────────────────────────────────
PHASE 1 — Reference bounds
  Purpose : establish the ceiling (all data labeled) and floor (few labels,
            no SSL) that every semi-supervised result must sit between.

  Upper bound (use SSL4MIS fully_supervised script):
    python ../SSL4MIS/code/train_fully_supervised_3D.py --dataset LA \
           --labeled_num 79 --max_samples 79

  Lower bound:
    python train.py --method baseline --dataset LA --labelnum 4
    python train.py --method baseline --dataset LA --labelnum 8

──────────────────────────────────────────────────────────────────────────────
PHASE 2 — Boundary supervision on labeled data only
  Purpose : show that the dual-head boundary loss improves results even
            without any unlabeled data (pure supervised gain).

    python train.py --method surface --dataset LA --labelnum 4
    python train.py --method surface --dataset LA --labelnum 8

──────────────────────────────────────────────────────────────────────────────
PHASE 3 — Basic semi-supervised (EMA, no enhancements)
  Purpose : show the gain from adding unlabeled data via Mean Teacher.
            Compare against Phase 2 at same label count.

    python train.py --method surface_ema --dataset LA --labelnum 4
    python train.py --method surface_ema --dataset LA --labelnum 8

──────────────────────────────────────────────────────────────────────────────
PHASE 4 — Ablation of enhancement components (one at a time)
  Purpose : justify each flag individually.  Compare every row against
            Phase 3 (surface_ema with no enhancements).

  4a. Strong augmentation only:
    python train.py --method surface_ema --dataset LA --labelnum 4 \
           --strong_aug

  4b. Distance-weighted boundary only:
    python train.py --method surface_ema --dataset LA --labelnum 4 \
           --boundary_decay 10

  4c. Uncertainty masking only:
    python train.py --method surface_ema --dataset LA --labelnum 4 \
           --uncertainty_thresh 0.3

  4d. Topology loss only:
    python train.py --method surface_ema --dataset LA --labelnum 4 \

──────────────────────────────────────────────────────────────────────────────
PHASE 5 — Full proposed method (all enhancements)
  Purpose : the complete BoundarySEG contribution; compare against all
            baselines and against each ablation from Phase 4.

  With full unlabeled set:
    python train.py --method surface_ema --dataset LA --labelnum 4 \
           --strong_aug --boundary_decay 10 --uncertainty_thresh 0.3 \

  With minimal unlabeled (key experiment for data-efficiency thesis):
    python train.py --method surface_ema --dataset LA \
           --labelnum 4 --max_samples 8 \
           --strong_aug --boundary_decay 10 --uncertainty_thresh 0.3 \

──────────────────────────────────────────────────────────────────────────────
PHASE 6 — Geometric SDF variant  (LA, VERSE only — SDF available)
  Purpose : show that eikonal + normal alignment further improve boundary
            metrics (HD95, ASD) on top of the full method.

    python train.py --method surface_sdf --dataset LA --labelnum 4 \
           --lambda_eikonal 0.1 --lambda_normal 0.05

──────────────────────────────────────────────────────────────────────────────
PHASE 7 — Data-efficiency curve
  Purpose : core thesis experiment — sweep unlabeled count to show your
            method reaches target Dice with fewer unlabeled samples.
            Fix labelnum, vary max_samples: 4+4, 4+8, 4+16, 4+full.

    for maxs in 8 12 20 79; do
      python train.py --method surface_ema --dataset LA \
             --labelnum 4 --max_samples $maxs \
             --strong_aug --boundary_decay 10 --uncertainty_thresh 0.3 \
    done

═══════════════════════════════════════════════════════════════════════════════
DATASETS
═══════════════════════════════════════════════════════════════════════════════

  LA        : 79 train / test  ·  binary  ·  (112,112,80)
  VERSE     : 190 train / 52 test  ·  binary  ·  (128,128,128)  SDF available
  ACDC      : 160 train / 40 val  ·  4-class  ·  (160,160,16)
              Note: surface_sdf now supports class-wise SDF targets on-the-fly for multiclass datasets.
                    Topology loss disabled (multi-class — binary Euler undefined).

═══════════════════════════════════════════════════════════════════════════════
QUICK REFERENCE — All flags
═══════════════════════════════════════════════════════════════════════════════

  Method:
    --method          baseline | surface | surface_ema | surface_sdf

  Data:
    --dataset         LA | VERSE | ACDC
    --labelnum N      number of labeled training cases
    --max_samples M   labeled + unlabeled total (default = full dataset)

  Enhancements (surface_ema):
    --strong_aug                    intensity + elastic augmentation on student
    --boundary_decay α              exp(-α·d) weight on CE loss (try α=10)
    --uncertainty_thresh t          mask consistency where teacher confidence < t
                                    (try t=0.3;  1.0 = disabled)

  Topology (all binary methods):

  Geometric (surface_sdf only):
    --lambda_eikonal w              enforce |∇SDF|=1  (try w=0.1)
    --lambda_normal  w              surface normal alignment (needs eikonal, try w=0.05)
    --lambda_sdf     w              SDF reconstruction weight (default 1.0)

  Infrastructure:
    --seed N          (run 1337, 42, 123 for all experiments)
    --gpu N
    --comet_key KEY   CometML API key (default: COMET_API_KEY env var)
"""

import os, sys, shutil, random, logging, argparse, json
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from kornia.morphology import erosion

from networks.vnet_sdf import VNet_mine, VNet_mine_SDF
from utils import ramps, losses

import sys as _sys, os as _os
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
# Prefer BoundarySEG-local modules, then shared helpers.
_sys.path.insert(0, _THIS_DIR)
_sys.path.insert(1, _os.path.join(_THIS_DIR, 'shared'))
from dataset_config import RESULTS_ROOT as _RESULTS_ROOT
from evaluation import evaluate_and_save, build_test_pairs as _build_test_pairs
from training_protocol import add_core_training_args, poly_lr, step_lr, cosine_lr, \
    add_deep_supervision_args, DeepSupervisionHeads, compute_ds_loss

from dataloaders.la_heart import (
    LAHeart, LAHeartSDF,
    RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler,
)
from dataloaders.acdc import ACDC, NUM_CLASSES as ACDC_NUM_CLASSES
from dataloaders.verse import VERSE

# ─────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True,
                    choices=['baseline', 'surface', 'surface_ema', 'surface_sdf'])
parser.add_argument('--dataset', default='LA', choices=['LA', 'VERSE', 'ACDC'])

# Dataset roots
from dataset_config import DATA_ROOT as _DATA_ROOT
parser.add_argument('--la_root',        default=_os.path.join(_DATA_ROOT, 'LA'))
parser.add_argument('--verse_root',     default=_os.path.join(_DATA_ROOT, 'VERSE'))
parser.add_argument('--acdc_root',      default=_os.path.join(_DATA_ROOT, 'ACDC'))
parser.add_argument('--save_root',     default=_RESULTS_ROOT)
parser.add_argument('--exp',           default='experiment')

# Training
parser.add_argument('--max_iterations', type=int,   default=6000)
parser.add_argument('--batch_size',     type=int,   default=4)
parser.add_argument('--labeled_bs',     type=int,   default=2,   help='Labeled samples per batch')
parser.add_argument('--labelnum',       type=int,   default=16,  help='Number of labeled training cases')
parser.add_argument('--max_samples',    type=int,   default=None, help='Total training samples (labeled + unlabeled). Default: full dataset')
parser.add_argument('--base_lr',        type=float, default=0.01)
parser.add_argument('--seed',           type=int,   default=1337)
parser.add_argument('--gpu',            type=str,   default='0')
parser.add_argument('--deterministic',  type=int,   default=1)

# Loss weights
parser.add_argument('--kernel_size',    type=int,   default=3,   help='Erosion kernel for boundary mask')
parser.add_argument('--erosion_mode',   type=str,   default='2d', choices=['2d', '3d'],
                     help="'2d' (default, backward-compatible): kornia erosion with a "
                          "(k,k) kernel applied per axial slice -- NOT true volumetric "
                          "erosion, since kornia treats the D axis as channels on a "
                          "(B,D,H,W) label tensor. '3d': genuine (k,k,k) cubic-structuring-"
                          "element erosion via 3D max-pooling on the inverted mask "
                          "(1 - maxpool3d(1-mask, k)). Requires odd kernel_size.")
parser.add_argument('--lambda_surface', type=float, default=1.0, help='Weight for boundary head loss')
parser.add_argument('--lambda_sdf',     type=float, default=1.0, help='Weight for SDF loss (surface_sdf, binary only)')
parser.add_argument('--ckpt_every', type=int, default=5000,
                    help='Write a diagnostic iter_<n>.pth every N iterations; 0 disables. '
                         'Nothing in this codebase reads these -- evaluation uses '
                         'last/best_loss/ema and there is no mid-training resume -- so they '
                         'are purely for inspecting a run mid-flight. The former default of '
                         '100 produced 200 files (~7 GB) per 20k run and 5.8 TB across the '
                         'results tree.')
parser.add_argument('--grad_diag_every', type=int, default=0,
                    help='If >0, every N iterations measure how the segmentation loss and the '
                         'auxiliary loss interact on the SHARED trunk (all params except the '
                         'out_conv* heads): the norm of each task gradient, their ratio, and '
                         'their cosine similarity. Negative cosine is the definition of gradient '
                         'conflict used by PCGrad (yu2020gradient), so this measures directly the '
                         'quantity the related-work paragraph asserts. Writes grad_diag.csv into '
                         'the snapshot dir. 0 = off (no cost, training path untouched). '
                         'surface_ema only.')
parser.add_argument('--band_edges', type=str, default='default', choices=['default', 'fine'],
                    help="Band edge set for --aux_head bands. 'default' = (2,5,10,20) voxels, "
                         "calibrated on LA. 'fine' = (1.5,2.5,3.5,5) for thin structures such "
                         "as the ACDC myocardium, where the default edges saturate.")
parser.add_argument('--n_bands', type=int, default=3,
                    help='Number of ordinal distance bands for --aux_head bands (max 4).')
parser.add_argument('--aux_head', type=str, default='boundary', choices=['boundary', 'sdf', 'bands', 'interface'],
                    help="What the AUXILIARY head regresses/classifies, for methods 'surface' and "
                         "'surface_ema'. 'boundary' (default) = binary contour classification "
                         "(sigmoid BCE + Dice) -- the \\name formulation. 'sdf' = signed-distance "
                         "REGRESSION (tanh head, truncated-L1 + surface + sign terms) -- the "
                         "SASSNet/DTC formulation. This is a strict controlled swap: identical "
                         "architecture, identical head capacity, identical --lambda_surface "
                         "weighting, identical EMA/uncertainty/copy-paste/flip machinery. ONLY the "
                         "auxiliary task's target and loss change, which is exactly the "
                         "classification-vs-regression claim under test.")

# Enhancement flags
parser.add_argument('--strong_aug',         action='store_true',
                    help='Strong augmentation on student input (surface_ema)')
parser.add_argument('--strong_aug_components', type=str, default='gamma,contrast,noise,elastic',
                    help='Comma list of strong_aug sub-transforms to apply '
                         '(subset of gamma,contrast,noise,elastic). Only used '
                         'when --strong_aug is set. Default = all four.')
parser.add_argument('--boundary_decay',     type=float, default=0.0,
                    help='α for exp(-α·d) boundary weighting; 0 = disabled')
parser.add_argument('--uncertainty_thresh', type=float, default=1.0,
                    help='Mask consistency where teacher uncertainty > thresh; 1.0 = disabled')

# Geometric SDF losses (surface_sdf + binary only)
parser.add_argument('--lambda_eikonal', type=float, default=0.0,
                    help='Weight for eikonal loss |∇SDF|=1 (surface_sdf only); 0 = disabled')
parser.add_argument('--eval_max_cases', type=int, default=0,
                    help='If >0, evaluate only first N test/val cases (useful for smoke runs).')
parser.add_argument('--lambda_normal',  type=float, default=0.0,
                    help='Weight for surface normal alignment (surface_sdf only). '
                         'Use together with --lambda_eikonal.')
# Uncertainty task weighting (Kendall et al., CVPR 2018)
parser.add_argument('--uncertainty_weights', action='store_true',
                    help='Learn per-task homoscedastic uncertainty weights (Kendall et al. 2018) '
                         'instead of using fixed --lambda_surface / --lambda_sdf. '
                         'Adds one log-σ² parameter per active auxiliary task (surf, sdf).')

# Copy-paste augmentation (labeled-only, no pseudo-labels)
parser.add_argument('--copy_paste',      action='store_true',
                    help='Labeled-only copy-paste: paste foreground from a random labeled '
                         'donor into each labeled sample in the batch (label union). '
                         'No pseudo-labels; operates within the labeled set only.')
parser.add_argument('--copy_paste_prob', type=float, default=0.5,
                    help='Per-sample probability of applying copy-paste (default 0.5).')

# Mixed precision
parser.add_argument('--amp', action='store_true',
                    help='Enable automatic mixed precision (torch.cuda.amp). '
                         'Disabled by default; pass --amp to activate.')

# EMA / consistency (surface_ema only)
parser.add_argument('--ema_decay',          type=float, default=0.99)
parser.add_argument('--consistency',             type=float, default=0.01)
parser.add_argument('--consistency_rampup',      type=float, default=40.0)
parser.add_argument('--consistency_type',        default='mse', choices=['mse', 'kl'])
parser.add_argument('--consistency_start_iter',  type=int,   default=0,
                    help='Iteration at which consistency loss switches on (0 = from start)')
parser.add_argument('--tta', action='store_true',
                    help='Test-time augmentation: average predictions over 8 axial flips')
parser.add_argument('--flip_axes', type=int, nargs='*', default=[],
                    help='Spatial dims to flip for equivariant consistency: '
                         '2=D (axial), 3=H (coronal), 4=W (sagittal). '
                         'Each axis adds one student forward pass per iteration.')
parser.add_argument('--flip_mode', type=str, default='both',
                    choices=['labeled', 'unlabeled', 'both'],
                    help='Which stream receives the flip loss: '
                         'labeled=supervised flip only, '
                         'unlabeled=equivariant consistency only, '
                         'both=both streams (default)')
parser.add_argument('--flip_sup_weight', type=float, default=1.0,
                    help='Weight for the labeled-stream flip supervised loss (default 1.0). '
                         'Set to 0.5 to keep total supervised contribution equal to baseline.')
parser.add_argument('--use_pseudo_labels', action='store_true',
                    help='Treat unlabeled cases as labeled using pre-generated pseudo_label h5 key')
parser.add_argument('--eval_only', action='store_true',
                    help='Skip training; load --resume_checkpoint and run evaluation once')
parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to a .pth checkpoint to load for --eval_only')

# Optional CometML
# Backbone
parser.add_argument('--backbone', default='vnet', choices=['vnet'],
                    help='vnet: original VNet_mine (default)  '
                         '(the only backbone used for any number in the paper).')
parser.add_argument('--normalization', default='batchnorm',
                    choices=['batchnorm', 'instancenorm', 'groupnorm'],
                    help='Normalization layer. '
                         'instancenorm avoids noisy BN statistics with small labeled batches.')
parser.add_argument('--label_smoothing', type=float, default=0.0,
                    help='Label smoothing ε for supervised CE loss (0 = disabled). '
                         'Prevents overconfidence on the small labeled set.')
parser.add_argument('--teacher_temp', type=float, default=1.0,
                    help='Temperature T for sharpening teacher logits before consistency loss '
                         '(logits /= T); T<1 sharpens, T=1 = disabled (default).')
# Optional CometML
parser.add_argument('--comet_key',     default=os.environ.get('COMET_API_KEY', ''),
                    help='CometML API key. Logging is OFF unless this is set '
                         '(via the flag or the COMET_API_KEY env var).')
parser.add_argument('--comet_project', default='semi-supervised-segmentation-new')

add_core_training_args(parser, default_seg_loss='ce')
add_deep_supervision_args(parser)
args = parser.parse_args()

_VALID_STRONG_AUG_COMPONENTS = {'gamma', 'contrast', 'noise', 'elastic'}
_strong_aug_components = {c.strip() for c in args.strong_aug_components.split(',') if c.strip()}
if args.strong_aug and not _strong_aug_components <= _VALID_STRONG_AUG_COMPONENTS:
    raise ValueError(f"--strong_aug_components must be a subset of "
                      f"{sorted(_VALID_STRONG_AUG_COMPONENTS)}, got {sorted(_strong_aug_components)}")

# ─────────────────────────────────────────────────────────────
# Dataset-level constants
# ─────────────────────────────────────────────────────────────
_NUM_CLASSES = {'LA': 2, 'VERSE': 2, 'ACDC': ACDC_NUM_CLASSES}
NUM_CLASSES   = _NUM_CLASSES[args.dataset]   # total including background
NUM_FG        = NUM_CLASSES - 1              # foreground channels = model output channels
IS_MULTICLASS = NUM_CLASSES > 2

# Shared evaluation helper expects args.root_path; map dataset-specific roots.
if not hasattr(args, 'root_path') or args.root_path is None:
    _ROOT_BY_DS = {
        'LA': args.la_root,
        'VERSE': args.verse_root,
        'ACDC': args.acdc_root,
    }
    args.root_path = _ROOT_BY_DS[args.dataset]

# Shared evaluation expects these canonical fields on args.
args.num_classes = NUM_CLASSES
args.patch_size = {
    'LA': (112, 112, 80),
    'VERSE': (128, 128, 128),
    'ACDC': (160, 160, 16),
}[args.dataset]
if args.method == 'surface_sdf' and args.backbone != 'vnet':
    raise ValueError('--method surface_sdf requires --backbone vnet (SDF head only exists on VNet_mine_SDF).')
if args.grad_diag_every > 0:
    if args.method != 'surface_ema':
        raise ValueError('--grad_diag_every is implemented for --method surface_ema only.')
    if args.amp:
        raise ValueError('--grad_diag_every with --amp would measure LOSS-SCALED gradients '
                         '(GradScaler multiplies the loss by a large dynamic factor), so the '
                         'norms and their ratio would be meaningless. Run the diagnostic without '
                         '--amp; the champion runs use amp=False anyway.')
if args.aux_head in ('sdf', 'bands', 'interface') and args.method not in ('surface', 'surface_ema'):
    raise ValueError("--aux_head sdf/bands is only meaningful for --method surface / surface_ema "
                     "(it swaps THAT auxiliary head's task). --method surface_sdf already has a "
                     "separate dedicated SDF head and is not a controlled swap.")

# ─────────────────────────────────────────────────────────────
# CometML (optional)
# ─────────────────────────────────────────────────────────────
experiment = None
if args.comet_key:
    from comet_ml import Experiment
    experiment = Experiment(api_key=args.comet_key, project_name=args.comet_project)
    experiment.add_tags([
        'BoundarySEG',
        args.dataset,
        args.method,
        getattr(args, 'backbone', 'vnet'),
        f'{args.labelnum}lbl',
        f'{max(0, (args.max_samples or args.labelnum) - args.labelnum)}unl',
    ])
    experiment.log_parameters(vars(args))
    experiment.log_parameter('Method', f'BoundarySEG_{args.method}')

def log_comet(name, val, step=None):
    if experiment is not None:
        experiment.log_metric(name, val, step)

# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
# Honor an externally-set CUDA_VISIBLE_DEVICES. Assigning it unconditionally
# would silently overwrite the caller's choice -- `CUDA_VISIBLE_DEVICES=1
# python train.py` would land on GPU 0, because --gpu defaults to '0' and
# args.gpu is the only device selector in this file (everything else just
# calls .cuda(), i.e. device 0 of whatever is visible).
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
if args.deterministic:
    cudnn.benchmark, cudnn.deterministic = False, True
else:
    cudnn.benchmark, cudnn.deterministic = True, False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

snapshot_path = os.path.join(
    args.save_root, args.dataset, args.method, args.exp,
    f'seed{args.seed}', f'label{args.labelnum}',
    f'kernel{args.kernel_size}', f'lam{args.lambda_surface}',
)

# ─────────────────────────────────────────────────────────────
# SDF loss  (surface_sdf, binary only)
# ─────────────────────────────────────────────────────────────
class SDFLoss:
    """L1 reconstruction + surface penalty + sign consistency + eikonal + normal alignment."""
    def __init__(self, truncation=20.0, w_recon=1.0, w_surf=0.5, w_sign=0.2,
                 narrow_band=10.0, w_eikonal=0.0, w_normal=0.0):
        self.trunc = truncation
        self.w_recon, self.w_surf, self.w_sign = w_recon, w_surf, w_sign
        self.w_eikonal = w_eikonal
        self.w_normal  = w_normal
        self.nb = narrow_band

    def __call__(self, pred, gt_sdf, gt_mask=None):
        # Multi-class: average per-class SDF losses over foreground channels.
        if pred.dim() == 5:
            C = pred.shape[1]
            total = pred.new_tensor(0.0)
            for c in range(C):
                cmask = None if gt_mask is None else gt_mask[:, c]
                total = total + self(pred[:, c], gt_sdf[:, c], cmask)
            return total / max(C, 1)

        pred = torch.clamp(pred, -self.trunc, self.trunc) / self.trunc
        gt   = torch.clamp(gt_sdf, -self.trunc, self.trunc) / self.trunc
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        gt   = torch.nan_to_num(gt,   nan=0.0, posinf=1.0, neginf=-1.0)
        surf_mask = (gt.abs() <= 0.5 / self.trunc).float()
        nb_mask = (gt.abs() <= self.nb / self.trunc).float()
        l_recon = ((pred - gt).abs() * nb_mask).sum() / nb_mask.sum().clamp_min(1.0)
        l_surf  = (pred.abs() * surf_mask).sum()       / surf_mask.sum().clamp_min(1.0)
        if gt_mask is not None:
            m = (gt_mask > 0).float()
            m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
            l_sign = F.binary_cross_entropy(torch.sigmoid(-pred / 0.5), m)
        else:
            l_sign = pred.new_tensor(0.)
        l_eikonal = eikonal_loss(pred)               if self.w_eikonal > 0 else pred.new_tensor(0.)
        l_normal  = normal_alignment_loss(pred, gt, surf_mask) \
                                                      if self.w_normal  > 0 else pred.new_tensor(0.)
        return (self.w_recon * l_recon + self.w_surf * l_surf
                + self.w_sign * l_sign + self.w_eikonal * l_eikonal
                + self.w_normal * l_normal)


def _sdf_gradients(sdf):
    """Central-difference gradients of (B, D, H, W), trimmed to interior."""
    dx = (sdf[:, 2:, :, :] - sdf[:, :-2, :, :])[:, :, 1:-1, 1:-1]
    dy = (sdf[:, :, 2:, :] - sdf[:, :, :-2, :])[:, 1:-1, :, 1:-1]
    dz = (sdf[:, :, :, 2:] - sdf[:, :, :, :-2])[:, 1:-1, 1:-1, :]
    return dx, dy, dz


def eikonal_loss(sdf):
    """mean((|∇φ| - 1)²) — constrains SDF head to be a valid distance function."""
    dx, dy, dz = _sdf_gradients(sdf)
    return ((dx**2 + dy**2 + dz**2).sqrt() - 1.0).pow(2).mean()


def normal_alignment_loss(pred_sdf, gt_sdf, surf_mask):
    """
    Angular loss between predicted and GT surface normals at boundary voxels.
    Uses arccos rather than cosine similarity — better gradient signal for
    severely misaligned normals (cosine similarity saturates near 180°).
    Requires eikonal loss to stabilise gradient directions off the surface.
    """
    dx_p, dy_p, dz_p = _sdf_gradients(pred_sdf)
    dx_g, dy_g, dz_g = _sdf_gradients(gt_sdf)
    mask = surf_mask[:, 1:-1, 1:-1, 1:-1]

    eps = 1e-6
    norm_p = (dx_p**2 + dy_p**2 + dz_p**2).sqrt().clamp_min(eps)
    norm_g = (dx_g**2 + dy_g**2 + dz_g**2).sqrt().clamp_min(eps)
    dot = ((dx_p/norm_p)*(dx_g/norm_g) + (dy_p/norm_p)*(dy_g/norm_g)
           + (dz_p/norm_p)*(dz_g/norm_g)).clamp(-1 + 1e-6, 1 - 1e-6)

    return (torch.acos(dot) * mask).sum() / mask.sum().clamp_min(1.0)


def build_classwise_sdf_targets(label_batch, num_classes):
    """Build class-wise signed distance maps and masks for foreground classes.

    Returns:
      sdf  : (B, C, D, H, W) float32, negative inside class region, positive outside
      mask : (B, C, D, H, W) float32, 1 inside class region else 0
    where C = num_classes - 1.
    """
    C = num_classes - 1
    if C <= 0:
        raise ValueError("num_classes must be >= 2")

    lbl_np = label_batch.detach().cpu().numpy().astype(np.int16)
    B, D, H, W = lbl_np.shape
    sdf_np = np.zeros((B, C, D, H, W), dtype=np.float32)
    msk_np = np.zeros((B, C, D, H, W), dtype=np.float32)

    for b in range(B):
        for c in range(1, num_classes):
            fg = (lbl_np[b] == c).astype(np.uint8)
            msk_np[b, c - 1] = fg
            if fg.any():
                sdf_np[b, c - 1] = edt(1 - fg) - edt(fg)
            else:
                # Class absent in this patch: keep a positive field so sign term has stable target.
                sdf_np[b, c - 1] = np.ones((D, H, W), dtype=np.float32)

    sdf = torch.from_numpy(sdf_np).to(label_batch.device)
    msk = torch.from_numpy(msk_np).to(label_batch.device)
    return sdf, msk

# ─────────────────────────────────────────────────────────────
# Strong augmentation  (surface_ema + --strong_aug)
# ─────────────────────────────────────────────────────────────
def elastic_deform(volume, alpha=0.12, sigma=5):
    """GPU elastic deformation via F.grid_sample + avg_pool3d smoothing."""
    B, C, D, H, W = volume.shape
    theta = torch.eye(3, 4, device=volume.device).unsqueeze(0).expand(B, -1, -1)
    grid  = F.affine_grid(theta, [B, C, D, H, W], align_corners=True)
    disp  = torch.randn(B, 3, D, H, W, device=volume.device) * alpha
    k     = max(3, int(sigma) * 2 + 1)
    disp  = F.avg_pool3d(disp, kernel_size=k, stride=1, padding=k // 2)
    disp  = disp.permute(0, 2, 3, 4, 1)
    return F.grid_sample(volume, (grid + disp).clamp(-1, 1),
                         mode='bilinear', padding_mode='reflection', align_corners=True)


def strong_augment(volume, components=('gamma', 'contrast', 'noise', 'elastic')):
    """
    Per-sample gamma + contrast + noise, then batch-wise elastic deformation.

    components: subset of {'gamma', 'contrast', 'noise', 'elastic'} to apply
    (default: all four). Lets callers isolate/drop individual sub-transforms,
    e.g. for per-component ablation.
    """
    result = []
    for b in range(volume.shape[0]):
        v = volume[b].clamp(min=1e-6)
        if 'gamma' in components:
            v = v ** random.uniform(0.7, 1.5)
        if 'contrast' in components:
            v = v *  random.uniform(0.8, 1.2)
        if 'noise' in components:
            v = v +  torch.randn_like(v) * 0.05
        result.append(v)
    out = torch.stack(result)
    if 'elastic' in components:
        out = elastic_deform(out)
    return out.clamp(0, 1)

# ─────────────────────────────────────────────────────────────
# Distance-weighted CE  (--boundary_decay)
# ─────────────────────────────────────────────────────────────
def boundary_weights(label_batch, alpha, num_classes):
    """
    Distance-decay weights exp(-alpha * d / d_max).

    Binary: distance to foreground/background interface.
    Multi-class: per-voxel distance to the boundary of that voxel's own class.

    label_batch: (B, D, H, W) integer labels.
    returns:    (B, D, H, W) float weights.
    """
    lbl_np = label_batch.detach().cpu().numpy().astype(np.int16)
    B, D, H, W = lbl_np.shape
    weights = np.ones((B, D, H, W), dtype=np.float32)

    if num_classes <= 2:
        binary_np = (lbl_np > 0).astype(np.uint8)
        for b in range(B):
            m = binary_np[b]
            if m.any() and not m.all():
                d = edt(m) + edt(1 - m)
                weights[b] = np.exp(-alpha * d / (d.max() or 1.0))
        return torch.from_numpy(weights).to(label_batch.device)

    # Multiclass: compute class-specific boundary distance and assign by GT class id.
    for b in range(B):
        lbl = lbl_np[b]
        for c in range(num_classes):
            m = (lbl == c).astype(np.uint8)
            if m.any() and not m.all():
                d = edt(m) + edt(1 - m)
                w_c = np.exp(-alpha * d / (d.max() or 1.0))
                weights[b][lbl == c] = w_c[lbl == c]
    return torch.from_numpy(weights).to(label_batch.device)


def weighted_bce(logits, target, weights):
    """BCE with per-voxel weights. All inputs: (B, D, H, W)."""
    raw = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)

# ─────────────────────────────────────────────────────────────
# Loss helpers  (unified binary + multi-class)
# ─────────────────────────────────────────────────────────────
def seg_loss(logits, label, num_classes, bw=None, label_smoothing=0.0):
    """
    Binary   (num_classes=2): sigmoid + BCE + Dice on the single output channel.
    Multi-class (num_classes>2): softmax + CrossEntropy + mean per-class Dice.

    logits : (B, C, D, H, W)  C = num_classes - 1 foreground channels
    label  : (B, D, H, W)     integer labels 0..num_classes-1
    bw     : (B, D, H, W)     distance weights (binary only, ignored for multi-class)
    label_smoothing : ε — smooths CE targets; Dice always uses hard labels.
    """
    C = num_classes - 1

    if C == 1:
        # ── Binary ──────────────────────────────────────────────────────
        lgt  = logits[:, 0]
        tgt  = (label == 1).float()
        soft = torch.sigmoid(lgt)
        # Smooth CE target: 0 → ε/2, 1 → 1 - ε/2
        tgt_ce = tgt * (1.0 - label_smoothing) + 0.5 * label_smoothing if label_smoothing > 0 else tgt
        if bw is not None:
            raw = F.binary_cross_entropy_with_logits(lgt, tgt_ce, reduction='none')
            l_ce = (raw * bw).sum() / bw.sum().clamp_min(1.0)
        else:
            l_ce = F.binary_cross_entropy_with_logits(lgt, tgt_ce)
        return l_ce + losses.dice_loss(soft, tgt)

    # ── Multi-class ──────────────────────────────────────────────────────
    bg          = torch.zeros(logits.shape[0], 1, *logits.shape[2:],
                              device=logits.device)
    full_logits = torch.cat([bg, logits], dim=1)
    l_ce = F.cross_entropy(full_logits, label.long(), reduction='none',
                           label_smoothing=label_smoothing)
    if bw is not None:
        l_ce = (l_ce * bw).sum() / bw.sum().clamp_min(1.0)
    else:
        l_ce = l_ce.mean()
    probs  = F.softmax(full_logits, dim=1)
    l_dice = sum(losses.dice_loss(probs[:, c], (label == c).float())
                 for c in range(1, num_classes)) / C
    return l_ce + l_dice


def _patch_interior_mask(label_batch, margin):
    """
    Returns a (B, D, H, W) boolean mask that is False within `margin` voxels
    of any patch face.  Boundaries detected in that strip are crop artifacts
    (the label is 1 right at the edge because the patch cut through the organ),
    not real anatomical surfaces.
    """
    B, D, H, W = label_batch.shape
    mask = torch.ones((B, D, H, W), dtype=torch.bool, device=label_batch.device)
    mask[:, :margin,  :,      :     ] = False
    mask[:, -margin:, :,      :     ] = False
    mask[:, :,        :margin, :    ] = False
    mask[:, :,        -margin:, :   ] = False
    mask[:, :,        :,       :margin ] = False
    mask[:, :,        :,       -margin:] = False
    return mask


def _erode3d(mask, kernel_size):
    """
    True (k,k,k) cubic-structuring-element erosion of a binary (B, D, H, W)
    mask, via max-pooling the inverted mask: erosion(A) = NOT(dilation(NOT(A))),
    and dilation-by-a-cube == max_pool3d with that cube as the window. Unlike
    kornia.morphology.erosion (2D, per-slice) this treats D as a real spatial
    axis, not a channel axis. Requires odd kernel_size for shape-preserving
    'same' padding (padding=k//2, stride=1); even k would silently grow the
    output by one voxel per axis, so we assert against it rather than let
    that pass quietly.
    """
    assert kernel_size % 2 == 1, f'--erosion_mode 3d requires odd kernel_size, got {kernel_size}'
    inv = 1.0 - mask.unsqueeze(1)                    # (B, 1, D, H, W)
    inv = F.max_pool3d(inv, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (1.0 - inv).squeeze(1)                     # (B, D, H, W), still {0,1}


def _surface6(mask):
    """Minimal 1-voxel inner surface: organ voxels with any non-organ 6-neighbour.

    This is the THINNEST shell the formulation admits. Erosion with a k=1
    structuring element is the identity, so XOR(L, Gamma_1(L)) is identically
    ZERO -- the k=1 row of the kernel ablation is therefore an ablation of the
    boundary head, not a thin-shell setting. _surface6 fills that gap with a
    genuine one-voxel surface.
    """
    p = F.pad(mask.unsqueeze(1), (1, 1, 1, 1, 1, 1), mode='replicate')
    nb = torch.stack([p[:, :, 2:, 1:-1, 1:-1], p[:, :, :-2, 1:-1, 1:-1],
                      p[:, :, 1:-1, 2:, 1:-1], p[:, :, 1:-1, :-2, 1:-1],
                      p[:, :, 1:-1, 1:-1, 2:], p[:, :, 1:-1, 1:-1, :-2]], 0)
    interior = nb.min(0).values.squeeze(1)
    return (mask.bool() & ~interior.bool()).float()


def shell_ratio(label_batch, num_classes, kernel_size=3, erosion_mode='2d'):
    """Per-class |shell| / |organ| for a batch of labels -- the diagnostic that
    says whether the boundary task is actually complementary to segmentation.

    Measured on the training labels: LA ~0.15, ACDC LV ~0.28, but ACDC RV ~0.62
    and myocardium ~0.73. Where the ratio approaches 1 the auxiliary head is
    largely re-predicting the primary mask rather than learning a surface, which
    is the regime in which the boundary head stops contributing.
    """
    out = {}
    for c in range(1, num_classes):
        m = (label_batch == c).float() if num_classes > 2 else (label_batch > 0).float()
        if m.sum() < 1:
            continue
        e = _erode3d(m, kernel_size) if erosion_mode == '3d' else \
            erosion(m, torch.ones(kernel_size, kernel_size, device=m.device))
        b = torch.logical_xor(m.bool(), e.bool()).float()
        out[c] = (b.sum() / m.sum().clamp_min(1)).item()
    return out


def build_boundary(label_batch, kernel_size, num_classes, erosion_mode='2d'):
    """
    Derive per-class boundary masks via erosion + XOR, then zero out any
    detected boundary within kernel_size voxels of every patch face.

    Those edge voxels are crop artifacts — when a patch slices through a large
    organ the label is 1 right at the cut, so erosion + XOR creates a false
    "boundary" along the crop edge.  Masking them prevents the surface head
    from learning phantom boundaries.

    erosion_mode='2d' (default): kornia erosion with a (k,k) kernel applied
      per axial slice -- backward-compatible with every prior result, but NOT
      true volumetric erosion (kornia treats the (B,D,H,W) label tensor's D
      axis as channels).
    erosion_mode='3d': genuine (k,k,k) cubic-structuring-element erosion via
      _erode3d() above.

    Returns (B, D, H, W) for binary, (B, C, D, H, W) for multi-class.
    """
    def _erode(mask_float):
        if erosion_mode == '3d':
            return _erode3d(mask_float, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size, device=mask_float.device)
        return erosion(mask_float, kernel)

    interior = _patch_interior_mask(label_batch, margin=kernel_size)
    C = num_classes - 1

    if C == 1:
        boundary = torch.logical_xor(label_batch,
                                     _erode(label_batch.float()).bool()).float()
        return boundary * interior.float()

    boundaries = []
    for c in range(1, num_classes):
        mask_c   = (label_batch == c).float()
        eroded_c = _erode(mask_c)
        b_c      = torch.logical_xor(mask_c.bool(), eroded_c.bool()).float()
        boundaries.append(b_c * interior.float())
    return torch.stack(boundaries, dim=1)          # (B, C, D, H, W)


def surface_loss(out_surf, boundary, num_classes, bw=None):
    """
    Boundary head loss: per-class sigmoid BCE + Dice, averaged.
    out_surf : (B, C, D, H, W)
    boundary : (B, D, H, W) binary for C=1, (B, C, D, H, W) for C>1
    bw       : (B, D, H, W) optional distance-decay weights (from boundary_weights)
    """
    C = num_classes - 1
    total = 0.0
    for c_idx in range(C):
        lgt_c = out_surf[:, c_idx]
        tgt_c = boundary if C == 1 else boundary[:, c_idx]
        soft_c = torch.sigmoid(lgt_c)
        if bw is not None:
            raw = F.binary_cross_entropy_with_logits(lgt_c, tgt_c.float(), reduction='none')
            l_bce = (raw * bw).sum() / bw.sum().clamp_min(1.0)
        else:
            l_bce = F.binary_cross_entropy_with_logits(lgt_c, tgt_c.float())
        total += l_bce + losses.dice_loss(soft_c, tgt_c.float())
    return total / C


# ─────────────────────────────────────────────────────────────
# Auxiliary-head task dispatch  (--aux_head boundary | sdf)
#
# This exists to TEST the regression-vs-classification claim, not to ship a
# second method. Both branches use the SAME head (VNet_mine.out_conv), the same
# --lambda_surface weight, and the same consistency/flip plumbing; only the
# target and the loss differ:
#
#   boundary : sigmoid(logits)      vs binary contour   -> BCE + Dice   (ours)
#   sdf      : tanh(logits)*trunc   vs signed distance  -> SDFLoss      (SASSNet/DTC)
#
# The SDF branch deliberately reuses the already-written SDFLoss rather than a
# fresh one, so the regression baseline is the same formulation the codebase
# already uses for --method surface_sdf.
# ─────────────────────────────────────────────────────────────
_SDF_TRUNC = 20.0
# Cumulative band edges in voxels for --aux_head bands. Band k covers voxels
# inside the organ within _BAND_EDGES[k] of the surface, so band 0 is the
# tightest shell and later bands nest outward.
#
# Measured coverage on real LA atria (fraction of organ):
#   <=2 vox 0.180 | <=5 vox 0.494 | <=10 vox 0.790 | <=20 vox 0.995
# Band 0 (0.180) is close to the deployed boundary target (0.145), so K=3 is
# "the boundary head plus two coarser distance channels" -- which is what makes
# the comparison against --aux_head boundary interpretable. The 20-voxel band is
# degenerate on LA (covers 99.5% of the organ) and should not be used.
_BAND_EDGES = (2.0, 5.0, 10.0, 20.0)
# Finer edges for thin structures (--band_edges fine). Half-integer on purpose:
# band k tests (d < 0) & (d > -t), and a voxel adjacent to background has
# d exactly -1, so t=1.0 yields an EMPTY band -- the same degeneracy as erosion
# r=1. Coverage with these edges: LA 0.16/0.28/0.37, ACDC LV 0.31/0.52/0.67,
# RV 0.66/0.90/0.98, Myo 0.81/0.98/1.00.
_BAND_EDGES_FINE = (1.5, 2.5, 3.5, 5.0)
# The auxiliary head has K channels under --aux_head bands but conf_mask is
# shaped like out_seg (NUM_FG channels). Broadcasting would make cons_surf K
# times larger than in the boundary arm; divide it back so the consistency
# weight means the same thing across arms.
_AUX_CH_SCALE = 1.0
_aux_sdf_criterion = None      # set in main() once args are known


def aux_sup_loss(out_aux, boundary, label_slice, num_classes, bw, sdf_target=None):
    """Supervised auxiliary loss for whichever task --aux_head selects.

    out_aux     : (B, n_classes, D, H, W) raw logits from the auxiliary head
    boundary    : binary contour target (used only when aux_head == 'boundary')
    label_slice : (B, D, H, W) integer labels (used to build the SDF target)
    sdf_target  : optional precomputed (sdf, mask) tuple, to avoid recomputing
                  the EDT for every flip axis (SDF is flip-equivariant, so the
                  caller flips the target instead of rebuilding it).
    """
    if args.aux_head == 'boundary':
        return surface_loss(out_aux, boundary, num_classes, bw)

    C = num_classes - 1

    if args.aux_head == 'interface':
        tgt = build_interface_targets(label_slice, num_classes)
        total = out_aux.new_tensor(0.0)
        for c in range(C):
            lgt = out_aux[:, c]
            total = total + F.binary_cross_entropy_with_logits(lgt, tgt[:, c]) \
                          + losses.dice_loss(torch.sigmoid(lgt), tgt[:, c])
        return total / max(C, 1)

    if args.aux_head == 'bands':
        # ORDINAL DISTANCE BANDS -- decouples the two things the boundary/SDF
        # comparison confounds. The boundary target is binary AND spatially
        # local; the SDF target is continuous AND carries long-range distance.
        # Bands are CLASSIFICATION-framed (same Dice+BCE loss as the boundary
        # head) but still encode distance, which fills the missing cell:
        #
        #                  local/binary        distance information
        #   classification  boundary head   ->  BANDS (this)
        #   regression      --                  SDF head
        #
        # Target for band k is the CUMULATIVE mask {inside and within t_k voxels
        # of the surface}. Cumulative (nested) rather than disjoint rings keeps
        # each channel a balanced binary problem and makes the ordering explicit.
        if sdf_target is None:
            sdf_target = build_classwise_sdf_targets(label_slice, num_classes)
        tgt_sdf, _ = sdf_target                      # (B, C, D, H, W), negative inside
        K = args.n_bands
        _edges = _BAND_EDGES_FINE if args.band_edges == 'fine' else _BAND_EDGES
        total = out_aux.new_tensor(0.0)
        for c in range(C):
            d = tgt_sdf[:, c]                        # signed distance for this class
            for k, t in enumerate(_edges[:K]):
                # inside the organ and within t voxels of the surface
                tgt_k = ((d < 0) & (d > -float(t))).float()
                lgt_k = out_aux[:, c * K + k]
                total = total + F.binary_cross_entropy_with_logits(lgt_k, tgt_k) \
                              + losses.dice_loss(torch.sigmoid(lgt_k), tgt_k)
        return total / max(C * K, 1)
    if sdf_target is None:
        sdf_target = build_classwise_sdf_targets(label_slice, num_classes)
    tgt_sdf, tgt_mask = sdf_target
    # tanh -> [-1,1], rescaled to voxel units so SDFLoss's internal
    # clamp(+-trunc)/trunc recovers exactly the normalised field.
    pred = torch.tanh(out_aux[:, :C].float()) * _SDF_TRUNC
    return _aux_sdf_criterion(pred, tgt_sdf, gt_mask=tgt_mask)


def grad_diag(model, l_seg, l_aux, lam, iter_num, out_csv):
    """Measure segmentation-vs-auxiliary gradient interaction on the shared trunk.

    Tests the related-work claim that a classification auxiliary head shares
    'gradient dynamics' with the segmentation head while an SDF REGRESSION head
    does not. Three quantities, all on the shared parameters only (everything
    except the out_conv* heads -- the heads are task-private, so gradients there
    cannot conflict by construction):

      grad_norm_seg / grad_norm_aux : the SCALE-MISMATCH claim. Two losses on
          wildly different scales push the trunk with wildly different force.
      cos(g_seg, g_aux)             : the CONFLICT claim. This is exactly
          PCGrad's definition (yu2020gradient) -- negative cosine means the two
          tasks pull the shared trunk in opposing directions.
      ratio over training           : the DYNAMICS claim (is it stable or does
          one task progressively swamp the other?).

    l_aux is the RAW auxiliary loss, before --lambda_surface. Gradient is linear
    in that scalar, so the effective weighted norm is lam * grad_norm_aux and
    needs no extra backward pass.

    Costs two extra backwards through the trunk, only on diagnostic iterations.
    Uses torch.autograd.grad, so nothing is accumulated into .grad and the real
    optimiser step is unaffected.
    """
    params = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith('out_conv')]
    g_seg = torch.autograd.grad(l_seg, params, retain_graph=True, allow_unused=True)
    g_aux = torch.autograd.grad(l_aux, params, retain_graph=True, allow_unused=True)
    v_seg = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                       for g, p in zip(g_seg, params)])
    v_aux = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                       for g, p in zip(g_aux, params)])
    n_seg = v_seg.norm().item()
    n_aux = v_aux.norm().item()
    cos = F.cosine_similarity(v_seg.unsqueeze(0), v_aux.unsqueeze(0)).item()

    new = not os.path.exists(out_csv)
    with open(out_csv, 'a') as f:
        if new:
            f.write('iter,aux_head,lambda,loss_seg,loss_aux_raw,'
                    'grad_norm_seg,grad_norm_aux_raw,grad_norm_aux_weighted,'
                    'ratio_weighted_over_seg,cosine\n')
        f.write('%d,%s,%g,%.6f,%.6f,%.6e,%.6e,%.6e,%.6e,%.6f\n'
                % (iter_num, args.aux_head, lam, l_seg.item(), l_aux.item(),
                   n_seg, n_aux, lam * n_aux,
                   (lam * n_aux) / max(n_seg, 1e-12), cos))
    return n_seg, n_aux, cos


def build_interface_targets(label_batch, num_classes):
    """Per-class INTER-CLASS interface: voxels of class c that touch a different
    foreground class (not background).

    Motivation: on ACDC the per-class shell target is 62-73% of the structure
    (myocardium is 2-3 voxels thick), so the boundary head largely re-predicts
    the segmentation mask. Interfaces between adjacent structures stay sparse
    even for thin objects, so they remain a genuinely complementary task.
    Multi-class only -- undefined for binary datasets.
    """
    C = num_classes - 1
    out = []
    for c in range(1, num_classes):
        m = (label_batch == c)
        other = (label_batch > 0) & (label_batch != c)
        d = F.max_pool3d(other.float().unsqueeze(1), kernel_size=3, stride=1,
                         padding=1).squeeze(1) > 0          # dilate the other classes
        out.append((m & d).float())
    return torch.stack(out, dim=1)                          # (B, C, D, H, W)


def conf_for_aux(conf_mask):
    """Match the uncertainty mask to the auxiliary head's channel count.

    conf_mask is shaped like out_seg (NUM_FG channels). The bands head emits
    C*K channels, laid out as [c0_b0, c0_b1, ..., c1_b0, ...], so each class's
    mask must be repeated K times -- repeat_interleave reproduces exactly that
    ordering. On binary data (NUM_FG=1) the old code broadcast 1 -> K silently;
    on multi-class it raised "size of tensor a (9) must match tensor b (3)".
    """
    if args.aux_head != 'bands':
        return conf_mask
    return conf_mask.repeat_interleave(args.n_bands, dim=1)


def aux_cons_field(x):
    """Map raw auxiliary logits to the space the consistency loss compares in.

    Boundary head: compare logits directly (matches the original behaviour).
    SDF head:      compare the bounded tanh field, which is what DTC does --
                   comparing raw pre-tanh logits would let the unbounded tail
                   dominate the MSE.

    All channels are kept (not just the C supervised ones) so the tensor still
    broadcasts against conf_mask, which is shaped like out_seg. The boundary
    branch already included the unsupervised trailing channel in cons_surf, so
    keeping it here preserves the control.
    """
    if args.aux_head in ('boundary', 'bands', 'interface'):
        return x                      # classification logits: compare directly
    return torch.tanh(x.float())      # sdf: bounded field, as DTC does


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def get_consistency_weight(iter_num):
    if iter_num < args.consistency_start_iter:
        return 0.0
    effective = iter_num - args.consistency_start_iter
    return args.consistency * ramps.sigmoid_rampup(effective // 150, args.consistency_rampup)

def update_ema(model, ema_model, alpha, step):
    alpha = min(1 - 1 / (step + 1), alpha)
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_(1 - alpha, p.data)

# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

def get_patch_size():
    return {
        'LA':        (112, 112, 80),
        'VERSE':     (128, 128, 128),
        'ACDC':      (160, 160, 16),
    }[args.dataset]


def build_model(has_dropout=True, ema=False):
    norm = args.normalization
    if args.method == 'surface_sdf':
        net = VNet_mine_SDF(n_channels=1, n_classes=NUM_FG,
                            normalization=norm, has_dropout=has_dropout)

    elif args.backbone == 'vnet':
        _naux = NUM_FG * args.n_bands if args.aux_head == 'bands' else None
        net = VNet_mine(n_channels=1, n_classes=NUM_FG, n_aux_classes=_naux,
                        normalization=norm, has_dropout=has_dropout)

    net = net.cuda()
    if ema:
        for p in net.parameters():
            p.detach_()
    return net

# ─────────────────────────────────────────────────────────────
# Dataset / dataloader factory
# ─────────────────────────────────────────────────────────────
def build_dataloader():
    patch_size = get_patch_size()

    if args.dataset == 'LA':
        root, max_samples = args.la_root, 79
        use_sdf = (args.method == 'surface_sdf')
        if use_sdf:
            db = LAHeartSDF(base_dir=root, split='train', transform=transforms.Compose([
                RandomRotFlip(with_sdf=True),
                RandomCrop(patch_size, with_sdf=True),
                ToTensor(sdf=True),
            ]))
        else:
            db = LAHeart(base_dir=root, split='train', transform=transforms.Compose([
                RandomRotFlip(), RandomCrop(patch_size), ToTensor(sdf=False),
            ]),
            use_pseudo_labels=getattr(args, 'use_pseudo_labels', False),
            labelnum=args.labelnum)

    elif args.dataset == 'VERSE':
        root, max_samples = args.verse_root, 197
        use_sdf = (args.method == 'surface_sdf')
        db = VERSE(base_dir=root, split='train', with_sdf=use_sdf,
                   transform=transforms.Compose([
                       RandomRotFlip(with_sdf=use_sdf),
                       ToTensor(sdf=use_sdf),
                   ]))

    elif args.dataset == 'ACDC':
        root, max_samples = args.acdc_root, 160
        db = ACDC(root=root, split='train', transform=transforms.Compose([
            RandomRotFlip(), RandomCrop(patch_size), ToTensor(sdf=False),
        ]))

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    if args.method == 'baseline':
        # Fully supervised: --method baseline never reads an unlabeled stream
        # (the loss only ever touches volume_batch[:lb]), so don't force the
        # TwoStreamBatchSampler's "pool must exceed labelnum" constraint --
        # that would make labelnum == full dataset size impossible to express.
        if args.labelnum > max_samples:
            raise ValueError(f"labelnum ({args.labelnum}) exceeds the dataset size ({max_samples}).")
        if args.labelnum < batch_size:
            raise ValueError(f"labelnum ({args.labelnum}) must be >= batch_size ({batch_size}) for --method baseline.")
        labeled_idxs = list(range(args.labelnum))
        subset = torch.utils.data.Subset(db, labeled_idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=4, pin_memory=True,
                            worker_init_fn=lambda wid: random.seed(args.seed + wid))
        return loader, root, db

    total_samples = max_samples if args.max_samples is None else min(max_samples, int(args.max_samples))
    if total_samples <= args.labelnum:
        raise ValueError(f"max_samples ({total_samples}) must be greater than labelnum ({args.labelnum}).")

    labeled_idxs   = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, total_samples))

    sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        batch_size, batch_size - args.labeled_bs,
    )
    loader = DataLoader(db, batch_sampler=sampler, num_workers=4, pin_memory=True,
                        worker_init_fn=lambda wid: random.seed(args.seed + wid))
    return loader, root, db


def build_test_pairs(root):
    """Return (image_path, label_path) pairs for evaluation."""
    if args.dataset == 'LA':
        with open(os.path.join(root, 'test.list')) as f:
            items = [l.strip() for l in f]
        paths = [os.path.join(root, '2018LA_Seg_Training Set', it, 'mri_norm2.h5')
                 for it in items]
        return [(p, None) for p in paths]   # label embedded in h5

    elif args.dataset == 'VERSE':
        with open(os.path.join(root, 'test.list')) as f:
            items = [l.strip() for l in f if l.strip()]
        paths = [os.path.join(root, 'data', it + '.h5') for it in items]
        return [(p, None) for p in paths]

    elif args.dataset == 'ACDC':  # use validation split (test has no labels)
        with open(os.path.join(root, 'dataset.json')) as f:
            meta = json.load(f)
        return [
            (os.path.join(root, e['image'].lstrip('./')),
             os.path.join(root, e['label'].lstrip('./')))
            for e in meta['validation']
        ]

    raise ValueError(f'Unknown dataset: {args.dataset}')

# ─────────────────────────────────────────────────────────────
# Uncertainty task weighting  (Kendall et al., CVPR 2018)
# ─────────────────────────────────────────────────────────────
class UncertaintyWeighter(torch.nn.Module):
    """
    Homoscedastic uncertainty weighting for N auxiliary tasks.

    Learns log(σ²) per task; the combined auxiliary loss is:
        Σ_i  L_i / (2 σ_i²)  +  log σ_i
      = Σ_i  L_i * exp(-log_var_i) * 0.5  +  log_var_i * 0.5

    Initialized to log_var = 0 (σ = 1, equal weights) so training
    starts identically to λ = 0.5 per task and diverges as needed.
    """
    def __init__(self, task_names):
        super().__init__()
        self.log_vars = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.zeros(1)) for name in task_names}
        )

    def weight(self, name, loss_val):
        lv = self.log_vars[name]
        return loss_val * torch.exp(-lv) * 0.5 + lv * 0.5

    def sigmas(self):
        return {name: float(torch.exp(lv * 0.5).item())
                for name, lv in self.log_vars.items()}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(snapshot_path, exist_ok=True)
    # Snapshot the source that produced this run, so a result can always be
    # traced back to the exact code. Copy the source tree EXPLICITLY -- a bare
    # copytree('.') of the repo root would follow the data/ symlinks (copying
    # entire datasets), duplicate checkpoints/, and recurse into results/, which
    # contains this very destination.
    code_dst = os.path.join(snapshot_path, 'code')
    if os.path.exists(code_dst):
        shutil.rmtree(code_dst)
    os.makedirs(code_dst)
    _src = _THIS_DIR
    _ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    for _d in ('networks', 'utils', 'dataloaders', 'shared'):
        _p = _os.path.join(_src, _d)
        if _os.path.isdir(_p):
            shutil.copytree(_p, _os.path.join(code_dst, _d), ignore=_ignore)
    for _f in ('train.py', 'test.py'):
        _p = _os.path.join(_src, _f)
        if _os.path.isfile(_p):
            shutil.copy2(_p, code_dst)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'log.txt'), level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(vars(args)))
    logging.info(f'dataset={args.dataset}  num_classes={NUM_CLASSES}  '
                 f'multiclass={IS_MULTICLASS}')

    # ── Models ──────────────────────────────────────────────
    model     = build_model()
    ema_model = build_model(ema=True) if args.method == 'surface_ema' else None
    ds_heads  = DeepSupervisionHeads(model, NUM_CLASSES).cuda() if args.deep_supervision else None

    # ── Data ────────────────────────────────────────────────
    trainloader, root, db = build_dataloader()
    patch_size = get_patch_size()
    logging.info(f'{len(trainloader)} iterations per epoch')

    model.train()
    if ema_model is not None:
        ema_model.train()

    # ── Uncertainty weighter (optional) ─────────────────────
    _uw_tasks = []
    if args.method in ('surface', 'surface_ema', 'surface_sdf'):
        _uw_tasks.append('surf')
    if args.method == 'surface_sdf':
        _uw_tasks.append('sdf')
    uw = UncertaintyWeighter(_uw_tasks).cuda() if args.uncertainty_weights and _uw_tasks else None

    def aux_loss(name, loss_val):
        """Apply fixed λ or learned uncertainty weight to one auxiliary loss."""
        if uw is not None:
            return uw.weight(name, loss_val)
        fixed = {'surf': args.lambda_surface, 'sdf': args.lambda_sdf}
        return fixed[name] * loss_val

    _params = list(model.parameters())
    if uw is not None:
        _params += list(uw.parameters())
        logging.info(f'Uncertainty weighting enabled for tasks: {_uw_tasks}')
    if ds_heads is not None:
        _params += list(ds_heads.parameters())
        logging.info('Deep supervision enabled (weights: x6=0.1, x7=0.2, x8=0.4)')
    optimizer = optim.SGD(_params, lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scaler    = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Consistency: sigmoid=True for binary (fixes softmax(1-channel)=1 bug),
    #              sigmoid=False (= softmax) for multi-class.
    _base_cons = (losses.softmax_mse_loss if args.consistency_type == 'mse'
                  else losses.softmax_kl_loss)
    if IS_MULTICLASS:
        consistency_criterion = _base_cons                          # softmax across classes
    else:
        consistency_criterion = lambda x, y: _base_cons(x, y, sigmoid=True)  # per-channel sigmoid

    sdf_criterion = (SDFLoss(w_eikonal=args.lambda_eikonal, w_normal=args.lambda_normal)
                     if args.method == 'surface_sdf' else None)

    # Criterion for the --aux_head sdf swap (SASSNet/DTC-style regression head).
    # Plain assignment: this block is module scope (inside `if __name__`), so it
    # rebinds the module-level name that aux_sup_loss reads.
    if args.aux_head == 'bands':
        _AUX_CH_SCALE = float(args.n_bands)
    if args.aux_head == 'sdf':
        _aux_sdf_criterion = SDFLoss(truncation=_SDF_TRUNC,
                                     w_eikonal=args.lambda_eikonal,
                                     w_normal=args.lambda_normal)

    writer   = SummaryWriter(os.path.join(snapshot_path, 'log'))
    iter_num = 0
    lr_      = args.base_lr
    best_train_loss = float('inf')
    best_loss_ckpt = os.path.join(snapshot_path, 'best_loss_model.pth')

    # Eval-only mode: skip training, load checkpoint, evaluate, exit.
    if getattr(args, 'eval_only', False):
        resume_path = getattr(args, 'resume_checkpoint', None)
        if not resume_path or not os.path.exists(resume_path):
            raise FileNotFoundError(f'--resume_checkpoint not found: {resume_path}')
        net_eval = build_model(has_dropout=False)
        net_eval.load_state_dict(torch.load(resume_path, map_location='cuda'))
        net_eval.eval()
        _pairs_eval = _build_test_pairs(args)
        evaluate_and_save(
            net_eval, _pairs_eval, args,
            os.path.join(snapshot_path, 'eval_tta'),
            baseline_name='BoundarySEG_tta',
            experiment=experiment,
            iteration=0,
        )
        sys.exit(0)

    for _ in tqdm(range(args.max_iterations // len(trainloader) + 1), ncols=70):
        for sampled_batch in trainloader:
            volume_batch = sampled_batch['image'].cuda()
            label_batch  = sampled_batch['label'].cuda()
            lb = args.labeled_bs

            # Labeled-only copy-paste: paste a random donor's foreground into each
            # labeled sample before supervised loss and boundary computation.
            if args.copy_paste:
                for i in range(lb):
                    if random.random() < args.copy_paste_prob:
                        donor = db[random.randint(0, args.labelnum - 1)]
                        d_img = donor['image'].to(volume_batch.device)   # (1,D,H,W)
                        d_lbl = donor['label'].to(label_batch.device)    # (D,H,W)
                        mask = (d_lbl > 0).float().unsqueeze(0)          # (1,D,H,W)
                        volume_batch[i] = volume_batch[i] * (1 - mask) + d_img * mask
                        # Label must follow the same paste mask as the image. torch.max
                        # only happens to be correct for binary {0,1} labels (where it's
                        # equivalent to logical OR); for multi-class labels it silently
                        # keeps the wrong class wherever the receiver's existing class ID
                        # is numerically larger than the donor's, decoupling image from
                        # label at those voxels.
                        label_batch[i]  = torch.where(mask.squeeze(0).bool(), d_lbl, label_batch[i])

            if args.method != 'baseline':
                boundary = build_boundary(label_batch, args.kernel_size, NUM_CLASSES,
                                           erosion_mode=args.erosion_mode)

            bw = (boundary_weights(label_batch[:lb], args.boundary_decay, NUM_CLASSES)
                  if args.boundary_decay > 0 else None)

            # ── Per-method forward + loss ──────────────────
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.method == 'baseline':
                    _, out_seg = model(volume_batch[:lb])
                    loss = seg_loss(out_seg, label_batch[:lb], NUM_CLASSES, label_smoothing=args.label_smoothing)
                    writer.add_scalar('loss/seg', loss.item(), iter_num)

                elif args.method == 'surface':
                    out_surf, out_seg = model(volume_batch[:lb])
                    l_seg  = seg_loss(out_seg, label_batch[:lb], NUM_CLASSES, bw, args.label_smoothing)
                    l_surf = aux_sup_loss(out_surf, boundary[:lb], label_batch[:lb],
                                          NUM_CLASSES, bw)
                    loss = l_seg + aux_loss('surf', l_surf)
                    writer.add_scalar('loss/seg',  l_seg.item(),  iter_num)
                    writer.add_scalar('loss/surf', l_surf.item(), iter_num)

                elif args.method == 'surface_ema':
                    # Teacher: full batch, weak aug (dataloader augments at load time)
                    with torch.no_grad():
                        teacher_surf, teacher_seg = ema_model(volume_batch)

                    # Sharpen teacher logits (T<1 → more confident predictions)
                    if args.teacher_temp != 1.0:
                        teacher_seg  = teacher_seg  / args.teacher_temp
                        teacher_surf = teacher_surf / args.teacher_temp

                    # Student: full batch, strong aug if enabled
                    student_in = (strong_augment(volume_batch, _strong_aug_components)
                                  if args.strong_aug else volume_batch)
                    out_surf, out_seg = model(student_in)

                    # Supervised on labeled portion.
                    # The SDF target is built ONCE per iteration and reused for every
                    # flip axis below (a flipped SDF is the SDF of the flipped label),
                    # so --aux_head sdf costs one EDT pass per iteration, not one per axis.
                    _sdf_tgt = (build_classwise_sdf_targets(label_batch[:lb], NUM_CLASSES)
                                if args.aux_head in ('sdf', 'bands') else None)
                    l_seg  = seg_loss(out_seg[:lb],  label_batch[:lb], NUM_CLASSES, bw, args.label_smoothing)
                    l_surf = aux_sup_loss(out_surf[:lb], boundary[:lb], label_batch[:lb],
                                          NUM_CLASSES, bw, sdf_target=_sdf_tgt)

                    if args.grad_diag_every > 0 and iter_num % args.grad_diag_every == 0:
                        _gn_s, _gn_a, _cos = grad_diag(
                            model, l_seg, l_surf, args.lambda_surface, iter_num,
                            os.path.join(snapshot_path, 'grad_diag.csv'))
                        writer.add_scalar('graddiag/norm_seg',  _gn_s, iter_num)
                        writer.add_scalar('graddiag/norm_aux',  _gn_a, iter_num)
                        writer.add_scalar('graddiag/cosine',    _cos,  iter_num)

                    # Consistency on full batch (labeled + unlabeled) with uncertainty mask
                    cw = get_consistency_weight(iter_num)
                    if args.uncertainty_thresh < 1.0:
                        if IS_MULTICLASS:
                            bg = torch.zeros(teacher_seg.shape[0], 1, *teacher_seg.shape[2:],
                                             device=teacher_seg.device)
                            p_full = F.softmax(torch.cat([bg, teacher_seg], dim=1), dim=1)
                            ent = -(p_full * torch.log(p_full + 1e-6)).sum(dim=1, keepdim=True)
                            ent = ent / np.log(NUM_CLASSES)
                            conf_mask = (ent < args.uncertainty_thresh).float().expand_as(out_seg)
                        else:
                            p = torch.sigmoid(teacher_seg)
                            conf_mask = (4.0 * p * (1.0 - p) < args.uncertainty_thresh).float()

                        denom     = conf_mask.sum().clamp_min(1.0)
                        cons_seg  = (consistency_criterion(out_seg,  teacher_seg)  * conf_mask).sum() / denom
                        _cm = conf_for_aux(conf_mask)
                        cons_surf = (consistency_criterion(aux_cons_field(out_surf),
                                                           aux_cons_field(teacher_surf)) * _cm).sum() / (_cm.sum().clamp_min(1.0))
                        cons_surf = cons_surf / _AUX_CH_SCALE
                        pct_conf  = conf_mask.mean().item()
                    else:
                        cons_seg  = consistency_criterion(out_seg,  teacher_seg).mean()
                        cons_surf = consistency_criterion(aux_cons_field(out_surf),
                                                          aux_cons_field(teacher_surf)).mean()
                        pct_conf  = 1.0

                    # ── Flip equivariance loss ────────────────────────────────
                    # For each axis: one student forward pass on the flipped full batch.
                    # flip_mode controls which streams receive the loss:
                    #   labeled   -> [:lb] supervised loss on flip(x), flip(label)
                    #   unlabeled -> equivariant consistency: student(flip(x)) vs flip(teacher(x))
                    #   both      -> both of the above
                    _flip_axes = getattr(args, 'flip_axes', [])
                    _flip_mode = getattr(args, 'flip_mode', 'both')
                    _do_lbl = _flip_mode in ('labeled', 'both')
                    _do_unl = _flip_mode in ('unlabeled', 'both')
                    if _flip_axes:
                        l_flip_sup  = volume_batch.new_tensor(0.0)
                        l_flip_cons = volume_batch.new_tensor(0.0)
                        for ax in _flip_axes:
                            # Single forward pass covers both labeled and unlabeled needs
                            in_f = volume_batch[:lb] if not _do_unl else volume_batch
                            vol_f = torch.flip(in_f, [ax])
                            out_surf_f, out_seg_f = model(vol_f)

                            if _do_lbl:
                                lbl_f   = torch.flip(label_batch[:lb], [ax - 1])
                                bnd_dim = ax - 1 if boundary.dim() == 4 else ax
                                bnd_f   = torch.flip(boundary[:lb], [bnd_dim])
                                bw_f    = torch.flip(bw, [ax - 1]) if bw is not None else None
                                # A flipped SDF is exactly the SDF of the flipped label
                                # (distance is isometry-invariant), so flip the cached
                                # target rather than recomputing the EDT per axis.
                                sdf_f = None if _sdf_tgt is None else (
                                    torch.flip(_sdf_tgt[0], [ax]), torch.flip(_sdf_tgt[1], [ax]))
                                # When unlabeled stream is also active, out_seg_f covers full
                                # batch so index [:lb]; otherwise it already is [:lb].
                                seg_slice = out_seg_f[:lb] if _do_unl else out_seg_f
                                sur_slice = out_surf_f[:lb] if _do_unl else out_surf_f
                                l_flip_sup = l_flip_sup \
                                    + seg_loss(seg_slice, lbl_f, NUM_CLASSES, bw_f, args.label_smoothing) \
                                    + aux_loss('surf', aux_sup_loss(sur_slice, bnd_f, lbl_f,
                                                                    NUM_CLASSES, bw_f,
                                                                    sdf_target=sdf_f))

                            if _do_unl:
                                t_seg_f  = torch.flip(teacher_seg,  [ax])
                                t_surf_f = torch.flip(teacher_surf, [ax])
                                if args.uncertainty_thresh < 1.0:
                                    conf_f  = torch.flip(conf_mask, [ax])
                                    denom_f = conf_f.sum().clamp_min(1.0)
                                    l_flip_cons = l_flip_cons \
                                        + (consistency_criterion(out_seg_f,  t_seg_f)  * conf_f).sum() / denom_f \
                                        + (consistency_criterion(aux_cons_field(out_surf_f),
                                                                 aux_cons_field(t_surf_f)) * conf_for_aux(conf_f)).sum() / conf_for_aux(conf_f).sum().clamp_min(1.0)
                                else:
                                    l_flip_cons = l_flip_cons \
                                        + consistency_criterion(out_seg_f,  t_seg_f).mean() \
                                        + consistency_criterion(aux_cons_field(out_surf_f),
                                                                aux_cons_field(t_surf_f)).mean()

                        l_flip_sup  = l_flip_sup  / len(_flip_axes)
                        l_flip_cons = l_flip_cons / len(_flip_axes)
                        writer.add_scalar('loss/flip_sup',  l_flip_sup.item(),  iter_num)
                        writer.add_scalar('loss/flip_cons', l_flip_cons.item(), iter_num)
                    else:
                        l_flip_sup  = volume_batch.new_tensor(0.0)
                        l_flip_cons = volume_batch.new_tensor(0.0)

                    _fsw = getattr(args, 'flip_sup_weight', 1.0)
                    loss = l_seg + aux_loss('surf', l_surf) \
                         + cw * (cons_seg + cons_surf) \
                         + _fsw * l_flip_sup + cw * l_flip_cons
                    update_ema(model, ema_model, args.ema_decay, iter_num)

                    writer.add_scalar('loss/seg',       l_seg.item(),     iter_num)
                    writer.add_scalar('loss/surf',      l_surf.item(),    iter_num)
                    writer.add_scalar('loss/cons_seg',  cons_seg.item(),  iter_num)
                    writer.add_scalar('loss/cons_surf', cons_surf.item(), iter_num)
                    writer.add_scalar('train/conf_pct',    pct_conf, iter_num)
                    writer.add_scalar('train/cons_weight', cw,       iter_num)

                elif args.method == 'surface_sdf':
                    out_surf, out_seg, out_sdf = model(volume_batch)
                    l_seg  = seg_loss(out_seg[:lb],  label_batch[:lb], NUM_CLASSES, bw, args.label_smoothing)
                    l_surf = surface_loss(out_surf[:lb], boundary[:lb], NUM_CLASSES, bw)

                    if NUM_CLASSES > 2:
                        tgt_sdf, tgt_mask = build_classwise_sdf_targets(label_batch[:lb], NUM_CLASSES)
                        l_sdf = sdf_criterion(out_sdf[:lb], tgt_sdf, gt_mask=tgt_mask)
                    else:
                        if 'sdf' in sampled_batch:
                            sdf_batch = sampled_batch['sdf'].cuda()
                            l_sdf = sdf_criterion(out_sdf[:lb, 0], sdf_batch[:lb],
                                                  gt_mask=(label_batch[:lb] > 0).float())
                        else:
                            tgt_sdf, tgt_mask = build_classwise_sdf_targets(label_batch[:lb], NUM_CLASSES)
                            l_sdf = sdf_criterion(out_sdf[:lb, 0], tgt_sdf[:, 0], gt_mask=tgt_mask[:, 0])

                    loss = l_seg + aux_loss('surf', l_surf) + aux_loss('sdf', l_sdf)
                    writer.add_scalar('loss/seg',  l_seg.item(),  iter_num)
                    writer.add_scalar('loss/surf', l_surf.item(), iter_num)
                    writer.add_scalar('loss/sdf',  l_sdf.item(),  iter_num)

                # ── Deep supervision (aux decoder heads) ──────────────
                if ds_heads is not None:
                    loss = loss + compute_ds_loss(
                        ds_heads, label_batch[:lb], NUM_CLASSES, args.seg_loss,
                        focal_gamma=args.focal_gamma, ds_weight=args.ds_weight)

            # ── Optimiser step ────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            _loss_val = float(loss.item())
            if _loss_val < best_train_loss:
                best_train_loss = _loss_val
                torch.save(model.state_dict(), best_loss_ckpt)
            iter_num += 1

            writer.add_scalar('loss/total', loss.item(), iter_num)
            log_comet('loss', loss.item(), iter_num)
            logging.info(f'iter {iter_num:05d} | loss={loss.item():.4f}')

            if args.lr_schedule == 'step':
                lr_ = step_lr(args.base_lr, iter_num, args.lr_step_size)
            elif args.lr_schedule == 'cosine':
                lr_ = cosine_lr(args.base_lr, iter_num, args.max_iterations)
            else:
                lr_ = poly_lr(args.base_lr, iter_num, args.max_iterations)
            # Linear LR warmup over the first lr_warmup_iters iterations. Damps the
            # early large steps where InstanceNorm + tiny multi-class batches drive
            # some seeds to diverge (ACDC 7lbl collapse). No-op when 0 (default).
            warmup = getattr(args, 'lr_warmup_iters', 0)
            if warmup > 0 and iter_num < warmup:
                lr_ = lr_ * float(iter_num + 1) / float(warmup)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            # Periodic checkpoints are diagnostic only: nothing in this codebase
            # reads iter_*.pth (evaluation uses last/best_loss/ema, and there is
            # no mid-training resume -- a preempted job restarts from 0). At the
            # old 100-iteration interval a single 20k run wrote 200 files / 7 GB,
            # which reached 5.8 TB across the results tree. Default is now 5000.
            if args.ckpt_every > 0 and iter_num % args.ckpt_every == 0:
                torch.save(model.state_dict(),
                           os.path.join(snapshot_path, f'iter_{iter_num}.pth'))

            # Sigma logging is cheap and stays at its original cadence.
            if iter_num % 100 == 0 and uw is not None:
                for name, sigma in uw.sigmas().items():
                    writer.add_scalar(f'uncertainty/sigma_{name}', sigma, iter_num)

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    # Ensure explicit final checkpoint exists irrespective of periodic save cadence.
    last_ckpt = os.path.join(snapshot_path, 'last_model.pth')
    torch.save(model.state_dict(), last_ckpt)
    if ema_model is not None:
        ema_ckpt = os.path.join(snapshot_path, 'ema_model.pth')
        torch.save(ema_model.state_dict(), ema_ckpt)
    writer.close()

    # ── Evaluation: best-train-loss and last ───────────────
    _pairs = _build_test_pairs(args)
    if int(getattr(args, 'eval_max_cases', 0)) > 0:
        _pairs = _pairs[:int(args.eval_max_cases)]
    net_best = build_model(has_dropout=False)
    if os.path.exists(best_loss_ckpt):
        net_best.load_state_dict(torch.load(best_loss_ckpt))
    else:
        net_best.load_state_dict(torch.load(last_ckpt))
    net_best.eval()
    evaluate_and_save(
        net_best,
        _pairs,
        args,
        os.path.join(snapshot_path, 'eval_best_loss'),
        baseline_name='BoundarySEG_best_loss',
        experiment=experiment,
        iteration=args.max_iterations,
    )

    net_last = build_model(has_dropout=False)
    net_last.load_state_dict(torch.load(last_ckpt))
    net_last.eval()
    evaluate_and_save(
        net_last,
        _pairs,
        args,
        os.path.join(snapshot_path, 'eval_last'),
        baseline_name='BoundarySEG_last',
        experiment=experiment,
        iteration=args.max_iterations,
    )

    if ema_model is not None:
        net_ema = build_model(has_dropout=False)
        net_ema.load_state_dict(torch.load(ema_ckpt))
        net_ema.eval()
        evaluate_and_save(
            net_ema,
            _pairs,
            args,
            os.path.join(snapshot_path, 'eval_ema'),
            baseline_name='BoundarySEG_ema',
            experiment=experiment,
            iteration=args.max_iterations,
        )
