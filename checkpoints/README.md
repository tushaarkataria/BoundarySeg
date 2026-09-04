# Released checkpoints

Twelve trained models: 3 seeds × 4 splits, ~37 MB each (~433 MB total).

Each file is the **last** training checkpoint (`last_model.pth`) of the run that
produced the corresponding paper row. No validation set was used and no
checkpoint was selected on test performance — the last iterate is what is
reported, for every method in the paper.

```
LA_4lbl_4unl/     seed1337.pth  seed42.pth  seed123.pth
LA_8lbl_4unl/     ...
VERSE_5lbl_5unl/  ...
VERSE_10lbl_5unl/ ...
```

Verify integrity with `sha256sum -c SHA256SUMS` from this directory.

The table below is what these files score on our hardware. On a different
GPU, cuDNN kernel choice shifts per-case Dice by up to ~1e-4, which is enough
to flip two cells in the last printed digit (LA 4:4 Dice -> 86.96, VerSe 10:5
HD95 -> 5.72). See the main README for the exact values.

## What each split scores

Mean ± std across the three seeds, from `scripts/test_checkpoints.sh`:

| Split | Dice | HD95 | ASD | Test cases |
|---|---|---|---|---|
| `LA_4lbl_4unl` | 86.95 ± 1.21 | 10.25 ± 2.61 | 2.87 ± 1.13 | 20 |
| `LA_8lbl_4unl` | 89.29 ± 0.47 | 8.58 ± 1.23 | 2.26 ± 0.43 | 20 |
| `VERSE_5lbl_5unl` | 83.73 ± 0.78 | 8.16 ± 0.81 | 2.55 ± 0.26 | 52 |
| `VERSE_10lbl_5unl` | 88.98 ± 0.27 | 5.73 ± 0.29 | 1.79 ± 0.10 | 52 |

## Loading one directly

The files are plain `state_dict`s for `networks.vnet_sdf.VNet_mine`. All four
splits are binary, so `n_classes=1` (foreground channels; background is an
implicit zero logit), and all four use InstanceNorm.

```python
import torch
from networks.vnet_sdf import VNet_mine

net = VNet_mine(n_channels=1, n_classes=1,
                normalization='instancenorm', has_dropout=False).cuda()
net.load_state_dict(torch.load('checkpoints/LA_4lbl_4unl/seed1337.pth'))
net.eval()
```

`forward()` returns both heads as a tuple, **auxiliary first**:

```python
boundary_logits, seg_logits = net(volume)   # out_conv, out_conv2
```

Only `seg_logits` is scored. It has one channel; background is an implicit zero
logit, so the mask is `seg_logits[:, 0] > 0`.

Loading with the wrong `normalization` succeeds silently and then predicts
garbage — it is the first thing to check if numbers come out far below the table.
