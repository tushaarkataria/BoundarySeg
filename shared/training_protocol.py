import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_core_training_args(parser: argparse.ArgumentParser, default_seg_loss: str = "ce_dice"):
    parser.add_argument(
        "--seg_loss",
        type=str,
        default=default_seg_loss,
        choices=["dice", "ce", "ce_dice", "bce_dice", "focal_dice"],
        help="Supervised segmentation loss for labeled data.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd"],
        help="Unified optimizer selection.",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="poly",
        choices=["poly", "step", "cosine"],
        help="Unified LR schedule. 'step' reproduces the legacy schedule "
             "(x0.1 every --lr_step_size iterations). 'cosine' applies "
             "cosine annealing from base_lr to 0 over max_iterations.",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=2500,
        help="Iterations between LR decays when --lr_schedule step.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Max grad norm for clip_grad_norm_. Set to <=0 to disable clipping.",
    )
    parser.add_argument(
        "--lr_warmup_iters",
        type=int,
        default=0,
        help="Linear LR warmup: scale base_lr from 0 -> full over the first N "
             "iterations (0 = disabled). Stabilizes the early steps where "
             "InstanceNorm + tiny multi-class batches otherwise diverge.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma for focal term when seg_loss=focal_dice.",
    )


def poly_lr(base_lr: float, iter_num: int, max_iterations: int, power: float = 0.9) -> float:
    return base_lr * (1.0 - float(iter_num) / float(max_iterations)) ** power


def vis_depth_slices(depth: int, n: int = 5):
    """Evenly spaced depth indices for TensorBoard image logging, over the middle
    50% of the volume. Baseline scripts hardcode `20:61:10` (5 slices, LA's 80-deep
    patch); on shallow patches like ACDC's 16 that range is empty and make_grid
    crashes on a 0-numel tensor. This scales proportionally to any depth."""
    n = min(n, depth)
    idx = torch.linspace(depth * 0.25, depth * 0.75, n).long().clamp(0, depth - 1)
    return idx.tolist()


def step_lr(base_lr: float, iter_num: int, step_size: int = 2500, gamma: float = 0.1) -> float:
    return base_lr * gamma ** (iter_num // step_size)


def cosine_lr(base_lr: float, iter_num: int, max_iterations: int) -> float:
    import math
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * iter_num / max_iterations))


def _dice_mean_fg_from_probs(probs: torch.Tensor, label: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 2:
        return 1.0 - (2.0 * torch.sum(probs[:, 1, ...] * (label == 1).float()) + 1e-8) / (
            torch.sum(probs[:, 1, ...] * probs[:, 1, ...]) + torch.sum((label == 1).float()) + 1e-8
        )
    dices = []
    for c in range(1, num_classes):
        tgt = (label == c).float()
        p = probs[:, c, ...]
        d = 1.0 - (2.0 * torch.sum(p * tgt) + 1e-8) / (torch.sum(p * p) + torch.sum(tgt) + 1e-8)
        dices.append(d)
    return torch.stack(dices).mean()


def _focal_ce_from_logits(logits: torch.Tensor, label: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, label.long(), reduction="none")
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** gamma) * ce
    return focal.mean()


def supervised_seg_loss(logits: torch.Tensor, label: torch.Tensor, num_classes: int, seg_loss: str, focal_gamma: float = 2.0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    dice = _dice_mean_fg_from_probs(probs, label, num_classes)

    if seg_loss == "dice":
        return dice
    if seg_loss == "ce":
        return F.cross_entropy(logits, label.long())
    if seg_loss == "ce_dice":
        ce = F.cross_entropy(logits, label.long())
        return 0.5 * (ce + dice)
    if seg_loss == "bce_dice":
        if num_classes > 2:
            raise ValueError("seg_loss=bce_dice is only supported for binary segmentation (num_classes=2).")
        fg_logit = logits[:, 1, ...]
        fg_tgt = (label > 0).float()
        bce = F.binary_cross_entropy_with_logits(fg_logit, fg_tgt)
        return 0.5 * (bce + dice)
    if seg_loss == "focal_dice":
        focal = _focal_ce_from_logits(logits, label, gamma=focal_gamma)
        return 0.5 * (focal + dice)
    raise ValueError(f"Unknown seg_loss: {seg_loss}")


# ─────────────────────────────────────────────────────────────────────────────
# Deep supervision
# ─────────────────────────────────────────────────────────────────────────────

# Decoder block names and their output channel multipliers (relative to n_filters=16).
# x6 = 1/8 resolution (128ch), x7 = 1/4 (64ch), x8 = 1/2 (32ch).
_DS_BLOCKS   = ['block_six', 'block_seven', 'block_eight']
_DS_CHANNELS = [128, 64, 32]
_DS_WEIGHTS  = [0.1, 0.2, 0.4]   # coarse → fine, sum = 0.7 << 1.0 (main head)


def add_deep_supervision_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--deep_supervision', action='store_true',
        help='Add auxiliary segmentation heads at x6 (1/8), x7 (1/4), x8 (1/2) decoder '
             'resolutions. Weights: 0.1 / 0.2 / 0.4 scaled by --ds_weight. '
             'Applied uniformly to all methods for a fair comparison.'
    )
    parser.add_argument(
        '--ds_weight', type=float, default=1.0,
        help='Global scale applied on top of the per-scale weights (0.1/0.2/0.4). '
             'Default 1.0 keeps the nnUNet-style weighting unchanged.'
    )


class DeepSupervisionHeads(nn.Module):
    """
    Auxiliary segmentation heads attached via forward hooks to the three
    intermediate VNet decoder blocks (block_six, block_seven, block_eight).

    Works with any VNet variant without modifying the network file.
    Hooks are automatically removed when this module is deleted.
    """
    def __init__(self, model: nn.Module, n_classes: int, n_filters: int = 16):
        super().__init__()
        channels = [n_filters * 8, n_filters * 4, n_filters * 2]
        self.heads  = nn.ModuleList([nn.Conv3d(c, n_classes, 1) for c in channels])
        self._feats = [None, None, None]
        self._hooks = []
        for i, block_name in enumerate(_DS_BLOCKS):
            block = getattr(model, block_name, None)
            if block is None:
                continue
            def _hook(j):
                def hook(m, inp, out):
                    self._feats[j] = out
                return hook
            self._hooks.append(block.register_forward_hook(_hook(i)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()

    def aux_logits(self):
        return [self.heads[i](f) for i, f in enumerate(self._feats) if f is not None]


def compute_ds_loss(
    ds_heads: DeepSupervisionHeads,
    label_labeled: torch.Tensor,
    num_classes: int,
    seg_loss: str,
    focal_gamma: float = 2.0,
    ds_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted deep supervision loss over the three auxiliary decoder heads.

    label_labeled : GT for the labeled portion only — shape (labeled_bs, D, H, W).
                    The aux logits are sliced to [:labeled_bs] automatically.
    """
    lb    = label_labeled.shape[0]
    total = label_labeled.new_tensor(0.)
    for logits_full, w in zip(ds_heads.aux_logits(), _DS_WEIGHTS):
        logits   = logits_full[:lb]
        lbl_down = F.interpolate(
            label_labeled.float().unsqueeze(1),
            size=logits.shape[2:],
            mode='nearest',
        ).squeeze(1).long()
        total = total + w * supervised_seg_loss(logits, lbl_down, num_classes, seg_loss, focal_gamma)
    return ds_weight * total
