"""
Unified post-training evaluation for all semi-supervised segmentation baselines.

Single entry point:
    evaluate_and_save(net, test_pairs, args, snapshot_path, baseline_name,
                      experiment=None, iteration=None)

What it does:
  1. Runs sliding-window inference on all test cases
  2. Computes per-class + mean metrics
  3. Prints per-class + mean to terminal
  4. Logs only mean metrics to CometML (if experiment is given)
  5. Saves a per-run CSV  →  <snapshot_path>/eval_results.csv
  6. Appends a summary row to a global CSV  →  <save_root>/all_results.csv

CSV schema
──────────
Per-run CSV (one row per case + a MEAN row):
  run_id, timestamp, baseline, dataset, model, method, exp, seed,
  labelnum, unlabelnum, patch_size, total_iterations, case_id,
  dice, jaccard, hd95, asd,                         ← binary (LA/VERSE)
  mean_dice, mean_hd95,                              ← multiclass summary (ACDC)
  dice_RV, dice_myocardium, dice_LV,                 ← ACDC per-class dice
  hd95_RV, hd95_myocardium, hd95_LV                  ← ACDC per-class HD95
  Binary rows leave multiclass columns empty; multiclass rows leave binary empty.

Global CSV (one row per run, means only):
  run_id, timestamp, baseline, dataset, model, method, exp, seed,
  labelnum, unlabelnum, mean_dice, mean_jaccard, mean_hd95, mean_asd
"""

import os
import csv
import math
import datetime
import h5py
import numpy as np
from medpy import metric
from skimage.measure import label as _cc_label
from tqdm import tqdm

# ── ACDC class names (3 foreground classes, indices 1-3) ────────────────────
ACDC_CLASS_NAMES = ['RV', 'myocardium', 'LV']

# Multiclass datasets are distinguished by num_classes (background + fg count).
_MC_CLASS_NAMES_BY_NUM_CLASSES = {
    4:  ACDC_CLASS_NAMES,
}


def _class_names_for(num_classes):
    if num_classes not in _MC_CLASS_NAMES_BY_NUM_CLASSES:
        raise ValueError(
            f'No class-name list registered for num_classes={num_classes}. '
            f'Known: {list(_MC_CLASS_NAMES_BY_NUM_CLASSES)}'
        )
    return _MC_CLASS_NAMES_BY_NUM_CLASSES[num_classes]


# ── All CSV column names ────────────────────────────────────────────────────
# Union of every registered multiclass dataset's class names, so the per-run
# and global CSVs share one fixed schema; rows from other datasets simply
# leave the unrelated columns blank.
_ALL_MC_CLASS_NAMES = list(ACDC_CLASS_NAMES)

_META_COLS = [
    'run_id', 'timestamp', 'baseline', 'dataset', 'model', 'method',
    'exp', 'seed', 'labelnum', 'unlabelnum', 'patch_size',
    'total_iterations', 'case_id',
]
_BINARY_COLS   = ['dice', 'jaccard', 'hd95', 'asd']
_MC_SUMMARY    = ['mean_dice', 'mean_hd95']
_MC_DICE_COLS  = [f'dice_{n}'  for n in _ALL_MC_CLASS_NAMES]
_MC_HD95_COLS  = [f'hd95_{n}'  for n in _ALL_MC_CLASS_NAMES]
_ALL_COLS      = _META_COLS + _BINARY_COLS + _MC_SUMMARY + _MC_DICE_COLS + _MC_HD95_COLS

_GLOBAL_COLS   = [
    'run_id', 'timestamp', 'baseline', 'dataset', 'model', 'method',
    'exp', 'seed', 'labelnum', 'unlabelnum',
    'mean_dice', 'mean_jaccard', 'mean_hd95', 'mean_asd',
]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(net, test_pairs, args, snapshot_path, baseline_name,
                      experiment=None, iteration=None):
    """
    Parameters
    ----------
    net           : trained model (any output format — seg head auto-detected)
    test_pairs    : list of (img_path, lbl_path_or_None)
                    lbl_path=None → label is embedded in the H5 file
    args          : argparse namespace
    snapshot_path : where to save predictions + per-run CSV
    baseline_name : e.g. 'BoundarySEG', 'UAMT', 'DTC'
    experiment    : CometML Experiment object, or None
    iteration     : checkpoint iteration, or None (uses args.max_iterations)
    """
    num_classes   = int(getattr(args, 'num_classes', 2))
    is_multiclass = num_classes > 2
    patch_size    = tuple(getattr(args, 'patch_size', (112, 112, 80)))

    test_save_dir = os.path.join(snapshot_path, 'test_predictions')
    os.makedirs(test_save_dir, exist_ok=True)
    
    # Optional smoke-mode cap for evaluation speed (e.g., EVAL_MAX_CASES=8).
    _cap = os.getenv('EVAL_MAX_CASES', '').strip()
    if _cap:
        try:
            _n = int(_cap)
            if _n > 0 and len(test_pairs) > _n:
                print(f"Evaluation case cap active: using {_n}/{len(test_pairs)} cases (EVAL_MAX_CASES)")
                test_pairs = test_pairs[:_n]
        except ValueError:
            print(f"Ignoring invalid EVAL_MAX_CASES={_cap!r}; expected positive integer.")

    meta  = _build_metadata(args, baseline_name, iteration, patch_size)

    tta = getattr(args, 'tta', False)
    if is_multiclass:
        rows = _run_multiclass(net, test_pairs, num_classes, patch_size, test_save_dir, tta=tta)
    else:
        rows = _run_binary(net, test_pairs, patch_size, test_save_dir, tta=tta)

    _print_results(rows, is_multiclass, num_classes)
    _log_comet(experiment, rows, is_multiclass, meta['run_id'], baseline_name)

    per_run_path  = os.path.join(snapshot_path, 'eval_results.csv')
    _write_per_run_csv(per_run_path, rows, meta, is_multiclass, num_classes)

    # Global aggregate CSV. Prefer save_root, but a finished run must never be
    # lost just because that root is unwritable (e.g. a stale shell script
    # pointing save_root at a read-only /home on the cluster). Fall back through
    # host-aware results_root -> snapshot_path (the per-run CSV already wrote
    # there, so it is known-writable).
    global_path = None
    for _root in (getattr(args, 'save_root', None),
                  getattr(args, 'results_root', None),
                  snapshot_path):
        if not _root:
            continue
        _cand = os.path.join(_root, 'all_results.csv')
        try:
            _append_global_csv(_cand, rows, meta, is_multiclass, num_classes)
            global_path = _cand
            break
        except OSError as e:
            print(f'WARNING: could not write global CSV to {_cand} ({e}); '
                  f'trying next fallback root.')
    if global_path is None:
        print('WARNING: global CSV could not be written to any root; '
              'per-run CSV is still saved.')

    print(f'\nPer-run CSV : {per_run_path}')
    print(f'Global CSV  : {global_path}')
    return rows


def build_test_pairs(args):
    """
    Build (img_path, lbl_path_or_None) pairs from args.dataset / args.root_path.
    Mirrors the logic in BoundarySEG/code/train.py::build_test_pairs.
    """
    import json
    root = args.root_path

    if args.dataset == 'LA':
        with open(os.path.join(root, 'test.list')) as f:
            items = [l.strip() for l in f if l.strip()]
        paths = [os.path.join(root, '2018LA_Seg_Training Set', it, 'mri_norm2.h5')
                 for it in items]
        return [(p, None) for p in paths]

    elif args.dataset == 'VERSE':
        with open(os.path.join(root, 'test.list')) as f:
            items = [l.strip() for l in f if l.strip()]
        paths = [os.path.join(root, 'data', it + '.h5') for it in items]
        return [(p, None) for p in paths]

    elif args.dataset == 'ACDC':
        with open(os.path.join(root, 'dataset.json')) as f:
            meta = json.load(f)
        return [
            (os.path.join(root, e['image'].lstrip('./')),
             os.path.join(root, e['label'].lstrip('./')))
            for e in meta['validation']
        ]

    raise ValueError(f"Unknown dataset: {args.dataset}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_window_single(net, image, patch_size, stride_xy, stride_z, num_fg):
    """Single-pass sliding-window inference. Returns soft score map (num_fg, W, H, D)."""
    import torch
    w, h, d = image.shape
    wpad = max(patch_size[0] - w, 0)
    hpad = max(patch_size[1] - h, 0)
    dpad = max(patch_size[2] - d, 0)
    if wpad or hpad or dpad:
        image = np.pad(image,
                       [(wpad//2, wpad-wpad//2),
                        (hpad//2, hpad-hpad//2),
                        (dpad//2, dpad-dpad//2)],
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z)  + 1

    score = np.zeros((num_fg,) + (ww, hh, dd), dtype=np.float32)
    cnt   = np.zeros((ww, hh, dd), dtype=np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                p  = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                t  = torch.from_numpy(p[np.newaxis, np.newaxis].astype(np.float32)).cuda()
                with torch.no_grad():
                    out = net(t)
                    if isinstance(out, (tuple, list)):
                        candidates = [x for x in out if hasattr(x, 'shape') and len(x.shape) == 5]
                        seg = None
                        if len(out) > 1 and hasattr(out[1], 'shape') and len(out[1].shape) == 5:
                            ch1 = int(out[1].shape[1])
                            if ch1 in (num_fg, num_fg + 1):
                                seg = out[1]
                        if seg is None:
                            seg = next((x for x in candidates if int(x.shape[1]) in (num_fg, num_fg + 1)), None)
                        if seg is None:
                            chs = [int(x.shape[1]) for x in candidates] if candidates else []
                            raise ValueError(
                                f'Could not find segmentation head in model outputs; '
                                f'candidate channels={chs}, expected {num_fg} or {num_fg + 1}'
                            )
                    else:
                        seg = out
                    ch = int(seg.shape[1])
                    if ch == num_fg:
                        y = torch.sigmoid(seg).cpu().numpy()[0]
                    elif ch == num_fg + 1:
                        y = torch.softmax(seg, dim=1)[:, 1:].cpu().numpy()[0]
                    else:
                        raise ValueError(
                            f'Unexpected network output channels: got {ch}, '
                            f'expected {num_fg} or {num_fg + 1} for num_fg={num_fg}'
                        )
                score[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score /= np.expand_dims(cnt, 0)

    # Unpad
    if wpad or hpad or dpad:
        wl, hl, dl = wpad//2, hpad//2, dpad//2
        score = score[:, wl:wl+w, hl:hl+h, dl:dl+d]

    return score


def _sliding_window(net, image, patch_size, stride_xy, stride_z, num_fg, tta=False):
    """Sliding-window inference with optional TTA (8 axial flips). Returns label map."""
    if tta:
        # Average scores over all 8 flip combinations (x, y, z axes)
        score = np.zeros((num_fg,) + image.shape, dtype=np.float32)
        for fx in (False, True):
            for fy in (False, True):
                for fz in (False, True):
                    aug = image
                    if fx: aug = aug[::-1]
                    if fy: aug = aug[:, ::-1]
                    if fz: aug = aug[:, :, ::-1]
                    aug = np.ascontiguousarray(aug)
                    s = _sliding_window_single(net, aug, patch_size, stride_xy, stride_z, num_fg)
                    if fz: s = s[:, :, :, ::-1]
                    if fy: s = s[:, :, ::-1]
                    if fx: s = s[:, ::-1]
                    score += np.ascontiguousarray(s)
        score /= 8.0
    else:
        score = _sliding_window_single(net, image, patch_size, stride_xy, stride_z, num_fg)

    if num_fg == 1:
        pred = (score[0] > 0.5).astype(np.int8)
    else:
        max_prob = score.max(axis=0)
        pred     = (score.argmax(axis=0) + 1).astype(np.int8)
        pred[max_prob < 0.5] = 0

    return pred


def _largest_connected_component(pred):
    """Keep only the largest connected component of a binary prediction.

    HD95/ASD are surface-distance metrics: a few stray false-positive voxels
    far from the main structure barely affect Dice/Jaccard but can dominate
    the distance statistics. Matches the post-processing used by the original
    SASSnet/BoundarySEG eval (test_util.py::getLargestCC).
    """
    if pred.sum() == 0:
        return pred
    labeled = _cc_label(pred)
    largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    return (labeled == largest).astype(pred.dtype)


def _run_binary(net, test_pairs, patch_size, save_dir, tta=False):
    """Evaluate binary segmentation on H5 files. Returns list of row dicts."""
    import nibabel as nib
    stride = _default_strides(patch_size)
    rows   = []
    ids    = _case_ids(test_pairs)

    for (img_path, _), case_id in zip(tqdm(test_pairs, desc='Evaluating'), ids):
        with h5py.File(img_path, 'r') as f:
            image = f['image'][:]
            gt    = f['label'][:]

        pred = _sliding_window(net, image, patch_size, stride, stride, num_fg=1, tta=tta)
        pred = _largest_connected_component(pred)

        if pred.sum() == 0:
            m = dict(dice=0.0, jaccard=0.0, hd95=0.0, asd=0.0)
        else:
            d, jc, hd, asd = _calc_binary(pred, gt)
            m = dict(dice=d, jaccard=jc, hd95=hd, asd=asd)

        rows.append({'case_id': case_id, **m})

        nib.save(nib.Nifti1Image(pred.astype(np.float32), np.eye(4)),
                 os.path.join(save_dir, f'{case_id}_pred.nii.gz'))
        nib.save(nib.Nifti1Image(gt.astype(np.float32), np.eye(4)),
                 os.path.join(save_dir, f'{case_id}_gt.nii.gz'))
        nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)),
                 os.path.join(save_dir, f'{case_id}_img.nii.gz'))

    return rows


def _run_multiclass(net, test_pairs, num_classes, patch_size, save_dir, tta=False):
    """Evaluate multiclass segmentation on NIfTI files. Returns list of row dicts."""
    import nibabel as nib
    num_fg      = num_classes - 1
    stride      = _default_strides(patch_size)
    class_names = _class_names_for(num_classes)
    rows        = []

    # Normalization must mirror the dataloader used at training time. ACDC has
    # no fixed unit scale (MRI), so it is normalized per-volume by percentile
    # clipping rather than by a fixed window.
    if num_classes == 4:
        from dataloaders.acdc import normalize_intensity as _normalize
    else:
        raise ValueError(f'No normalization registered for num_classes={num_classes}.')

    ids = _case_ids(test_pairs)
    for (img_path, lbl_path), case_id in zip(tqdm(test_pairs, desc='Evaluating'), ids):
        image = nib.load(img_path).get_fdata(dtype=np.float32)
        gt    = nib.load(lbl_path).get_fdata().astype(np.int16)
        image = _normalize(image)

        pred = _sliding_window(net, image, patch_size, stride, stride, num_fg=num_fg, tta=tta)

        dice_pc   = {}
        hd95_pc   = {}
        for c in range(1, num_classes):
            pc = (pred == c); gc = (gt == c)
            if gc.sum() == 0:
                dice_pc[c] = np.nan; hd95_pc[c] = np.nan
            elif pc.sum() == 0:
                dice_pc[c] = 0.0
                hd95_pc[c] = float(np.sqrt(sum(s**2 for s in pred.shape)))
            else:
                dice_pc[c] = metric.binary.dc(pc, gc)
                hd95_pc[c] = metric.binary.hd95(pc, gc)

        mean_dice = float(np.nanmean(list(dice_pc.values())))
        mean_hd95 = float(np.nanmean(list(hd95_pc.values())))

        row = {'case_id': case_id, 'mean_dice': mean_dice, 'mean_hd95': mean_hd95}
        for c_idx, name in enumerate(class_names):
            c = c_idx + 1
            row[f'dice_{name}'] = dice_pc.get(c, np.nan)
            row[f'hd95_{name}'] = hd95_pc.get(c, np.nan)
        rows.append(row)

        nib.save(nib.Nifti1Image(pred.astype(np.float32), np.eye(4)),
                 os.path.join(save_dir, f'{case_id}_pred.nii.gz'))
        nib.save(nib.Nifti1Image(gt.astype(np.float32), np.eye(4)),
                 os.path.join(save_dir, f'{case_id}_gt.nii.gz'))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Print / log / CSV
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(rows, is_multiclass, num_classes):
    print('\n' + '='*60)
    if is_multiclass:
        all_dice = [r['mean_dice'] for r in rows]
        all_hd   = [r['mean_hd95'] for r in rows]
        print(f'Mean Dice : {np.nanmean(all_dice):.4f}')
        print(f'Mean HD95 : {np.nanmean(all_hd):.2f}')
        print('\nPer-class (mean over cases):')
        for c_idx, name in enumerate(_class_names_for(num_classes)):
            c     = c_idx + 1
            dvals = [r.get(f'dice_{name}', np.nan) for r in rows]
            hvals = [r.get(f'hd95_{name}', np.nan) for r in rows]
            print(f'  {c:2d} {name:<20s}  dice={np.nanmean(dvals):.4f}  '
                  f'hd95={np.nanmean(hvals):.2f}')
    else:
        for key in ('dice', 'jaccard', 'hd95', 'asd'):
            vals = [r[key] for r in rows]
            print(f'{key:8s}: {np.mean(vals):.4f}')
    print('='*60)


def _log_comet(experiment, rows, is_multiclass, run_id, baseline_name=''):
    if experiment is None:
        return

    # Distinguish dual-eval runs in Comet for easy filtering/comparison.
    # Examples: mean_dice_best, mean_dice_last
    suffix = ''
    if baseline_name.endswith('_best_loss'):
        suffix = '_best'
    elif baseline_name.endswith('_last'):
        suffix = '_last'

    if is_multiclass:
        experiment.log_metric(f'mean_dice{suffix}', np.nanmean([r['mean_dice'] for r in rows]))
        experiment.log_metric(f'mean_hd95{suffix}', np.nanmean([r['mean_hd95'] for r in rows]))
    else:
        for key in ('dice', 'jaccard', 'hd95', 'asd'):
            experiment.log_metric(f'mean_{key}{suffix}', np.mean([r[key] for r in rows]))


def _write_per_run_csv(path, rows, meta, is_multiclass, num_classes=None):
    mean_row = _compute_mean_row(rows, is_multiclass, num_classes)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_COLS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({**meta, **row})
        # MEAN summary row at the bottom
        writer.writerow({**meta, 'case_id': 'MEAN', **mean_row})


def _append_global_csv(path, rows, meta, is_multiclass, num_classes=None):
    write_header = not os.path.exists(path)
    mean_row = _compute_mean_row(rows, is_multiclass, num_classes)

    # Map to global columns
    if is_multiclass:
        global_means = {
            'mean_dice':    mean_row.get('mean_dice', ''),
            'mean_hd95':    mean_row.get('mean_hd95', ''),
            'mean_jaccard': '',
            'mean_asd':     '',
        }
    else:
        global_means = {
            'mean_dice':    mean_row.get('dice', ''),
            'mean_jaccard': mean_row.get('jaccard', ''),
            'mean_hd95':    mean_row.get('hd95', ''),
            'mean_asd':     mean_row.get('asd', ''),
        }

    global_row = {k: meta.get(k, '') for k in _GLOBAL_COLS}
    global_row.update(global_means)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_GLOBAL_COLS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(global_row)


def _compute_mean_row(rows, is_multiclass, num_classes=None):
    if is_multiclass:
        mean = {
            'mean_dice': float(np.nanmean([r['mean_dice'] for r in rows])),
            'mean_hd95': float(np.nanmean([r['mean_hd95'] for r in rows])),
        }
        for name in _class_names_for(num_classes):
            mean[f'dice_{name}'] = float(np.nanmean([r.get(f'dice_{name}', np.nan) for r in rows]))
            mean[f'hd95_{name}'] = float(np.nanmean([r.get(f'hd95_{name}', np.nan) for r in rows]))
    else:
        mean = {k: float(np.mean([r[k] for r in rows])) for k in _BINARY_COLS}
    return mean


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_metadata(args, baseline_name, iteration, patch_size):
    labelnum   = int(getattr(args, 'labelnum',
                    getattr(args, 'labeled_num',
                    getattr(args, 'labeled_bs', 0))))
    max_samples = int(getattr(args, 'max_samples', None) or 0)
    unlabelnum  = max(0, max_samples - labelnum)

    total_iter  = iteration or getattr(args, 'max_iterations',
                                getattr(args, 'max_iteration', ''))

    model_name  = getattr(args, 'backbone',
                  getattr(args, 'model', 'vnet'))
    method      = getattr(args, 'method', '')
    exp         = getattr(args, 'exp', '')
    seed        = getattr(args, 'seed', '')
    dataset     = getattr(args, 'dataset', '')

    ts     = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f'{baseline_name}_{dataset}_{exp}_{labelnum}lbl_{seed}seed_{ts}'

    return {
        'run_id':            run_id,
        'timestamp':         ts,
        'baseline':          baseline_name,
        'dataset':           dataset,
        'model':             model_name,
        'method':            method,
        'exp':               exp,
        'seed':              seed,
        'labelnum':          labelnum,
        'unlabelnum':        unlabelnum,
        'patch_size':        'x'.join(str(s) for s in patch_size),
        'total_iterations':  total_iter,
    }


def _calc_binary(pred, gt):
    d   = metric.binary.dc(pred, gt)
    jc  = metric.binary.jc(pred, gt)
    hd  = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return d, jc, hd, asd


def _default_strides(patch_size):
    """Conservative strides: half patch size, minimum 4."""
    return max(4, min(patch_size) // 2)


def _stem(path):
    """Double-strip extension to handle .nii.gz and .h5."""
    name = os.path.basename(path)
    for _ in range(2):
        base, ext = os.path.splitext(name)
        if ext:
            name = base
    return name


def _case_ids(test_pairs):
    """Unique per-case identifiers for a list of (img_path, lbl_path) pairs.

    Normally the filename stem IS the case id (VERSE, ACDC). LA is
    different: every case lives at
        <root>/2018LA_Seg_Training Set/<CASE_ID>/mri_norm2.h5
    so the stem is the literal string 'mri_norm2' for ALL cases. That silently
    made every case share one id, which (a) overwrote each saved prediction with
    the next, leaving exactly ONE prediction volume per run and no record of
    which case it belonged to, and (b) filled the case_id column of
    eval_results.csv with a constant, so per-case pairing across methods could
    not be verified.

    Fix: fall back to the PARENT DIRECTORY name whenever the stems are not
    unique. This is a no-op for datasets whose filenames already differ, so it
    cannot change existing behaviour anywhere the ids were already correct.
    """
    stems = [_stem(p) for p, _ in test_pairs]
    if len(set(stems)) == len(stems):
        return stems

    parents = [os.path.basename(os.path.dirname(p)) for p, _ in test_pairs]
    if len(set(parents)) == len(parents):
        return parents

    # Last resort: disambiguate with an index so predictions never overwrite.
    return ['%s_%03d' % (s, i) for i, s in enumerate(stems)]
