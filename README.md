# BoundarySEG

*An Embarrassingly Simple Method To Boost Medical Image Segmentation Performance
for Low Data Regimes* — [arXiv:2505.09829](https://arxiv.org/abs/2505.09829)

Semi-supervised 3D medical image segmentation with a **boundary-classification
auxiliary head**. A single V-Net carries two output heads on a shared trunk: one
predicts the segmentation mask, the other predicts a thin boundary shell as a
*classification* target. A mean-teacher branch supplies consistency on unlabeled
volumes, and an uncertainty gate decides which of the teacher's voxels are
allowed to teach.

The method is aimed at the regime where labeled volumes are scarce **and the
unlabeled pool is small too** — where most semi-supervised methods fall below
their own labeled-only baseline. On VerSe with 5 labeled and 5 unlabeled volumes,
nine of the ten baselines we ran score *below* the labeled-only reference of
62.17 Dice; BoundarySEG is 21.6 above it.

This repository contains only what is needed to reproduce the paper's numbers.

---

## Results

Mean ± std over **3 seeds** (1337, 42, 123), from the **last** training
checkpoint — no validation set, no checkpoint selection. Dice in %, HD95 and ASD
in voxels. `L:U` = labeled : unlabeled volumes.

| Dataset | L:U | Dice | HD95 | ASD | Best baseline (Dice) |
|---|---|---|---|---|---|
| LA | 4:4 | **86.95 ± 1.21** | **10.25 ± 2.61** | **2.87 ± 1.13** | BCP 78.20 |
| LA | 8:4 | **89.29 ± 0.47** | **8.58 ± 1.23** | 2.26 ± 0.43 | MCNet 86.54 |
| VerSe | 5:5 | **83.73 ± 0.78** | **8.16 ± 0.81** | 2.55 ± 0.26 | UnCo 64.77 |
| VerSe | 10:5 | **88.98 ± 0.27** | 5.73 ± 0.29 | 1.79 ± 0.10 | BCP 87.09 |

Bold marks a lead significant at p < 0.05 under a paired Wilcoxon signed-rank
test over test cases (LA n=20, VerSe n=52), seed-averaged per case. Those tests
were run against the seven baselines available at submission; CAML, PPC and UnCo
were added afterwards and are not yet formally tested on VerSe, though the Dice
and HD95 margins over them are large.

Where a cell is not bold, the metric is not a clear lead: LA 8:4 HD95 ties MCF
(p=0.57), and VerSe 10:5 HD95/ASD sit inside a cluster — nominally SS-Net (4.93 /
1.40), UnCo (5.29 / 1.54) and BCP (— / 1.48) are ahead of ours on one or both,
with SS-Net and BCP previously tested as ties. Overlap is where the method wins
on VerSe 10:5; surface quality there is a wash.

Trained checkpoints for all four splits are in [`checkpoints/`](checkpoints/),
and `scripts/test_checkpoints.sh` reproduces this table without retraining.

**On reproducing these to the last digit.** Inference is deterministic, but cuDNN
picks different kernels on different GPU models, which moves per-case Dice by up
to ~1e-4. Two cells above sit within 3e-6 of a rounding boundary, so on hardware
other than ours they print one digit differently — LA 4:4 Dice as `86.96`
(86.955054 vs 86.954945) and VerSe 10:5 HD95 as `5.72` (5.724747 vs 5.725032).
Both were confirmed on an RTX 3090. Anything larger than the third decimal is a
real difference and worth reporting as a bug.

---

## Install

```bash
conda create -n boundaryseg python=3.9 && conda activate boundaryseg
pip install -r requirements.txt
```

PyTorch 2.1.2 + CUDA 12.1 is what every reported number was produced with.

**GPU memory:** LA needs ~9 GB (112×112×80 patches). VerSe trains on 128³
patches and needs a 24 GB card.

## Data

Symlink the datasets into `data/` — the code looks there by default:

```bash
ln -s /path/to/LA     data/LA
ln -s /path/to/VERSE  data/VERSE
```

[`data/README.md`](data/README.md) gives the expected directory layout for each
dataset and explains how the labeled/unlabeled split is formed.

---

## Evaluate the released checkpoints

```bash
bash scripts/test_checkpoints.sh                 # all four splits
bash scripts/test_checkpoints.sh LA_4lbl_4unl    # just one
```

This scores all three seeds of a split and prints mean ± std, matching the table
above up to the last-digit caveat noted there. For a single checkpoint:

```bash
python test.py --dataset LA --checkpoint checkpoints/LA_4lbl_4unl/seed1337.pth
```

Each run writes `eval_results.csv` — one row per test case plus a `MEAN` row —
next to the checkpoint. Per-case rows are what the significance tests consume. Download checkpoints from [here](https://drive.google.com/drive/folders/1Bqrkv_DffiWVkhCNBYOqJQFctY2QTvkE?usp=sharing)

**Comparing your own method?** `test.py` applies the same protocol to any
checkpoint with this repository's V-Net architecture. To compare a *different*
architecture on equal footing, reuse `shared/evaluation.py::evaluate_and_save`,
which is the single evaluation path every number in the paper went through —
sliding-window inference at the training patch size, largest-connected-component
post-processing on binary datasets, and metrics via `medpy`.

## Train from scratch

```bash
bash scripts/train_la.sh 4        # LA, 4 labeled / 4 unlabeled
bash scripts/train_la.sh 8        # LA, 8 labeled / 4 unlabeled
bash scripts/train_verse.sh 5     # VerSe, 5 labeled / 5 unlabeled
bash scripts/train_verse.sh 10    # VerSe, 10 labeled / 5 unlabeled
```

Each script runs all three seeds in sequence, then prints mean ± std. One seed is
about 4 h on an RTX 3090 (0.735 s/iteration × 20k) and roughly 5 h on a 12 GB
card, so a full sweep is 12–15 h; put the seeds on separate GPUs to parallelize:

```bash
SEEDS=1337 GPU=0 bash scripts/train_la.sh 4 &
SEEDS=42   GPU=1 bash scripts/train_la.sh 4 &
SEEDS=123  GPU=2 bash scripts/train_la.sh 4 &
```

Training evaluates its own final checkpoint automatically; results land in
`results/<DATASET>/surface_ema/<exp>/.../eval_last/eval_results.csv`. The summary
step globs all seeds of a split, so whichever run finishes last prints the
complete mean ± std table.

### What retraining actually gives you

LA 4:4 was retrained from scratch on an RTX 3090 — all three seeds, nothing
carried over from the released checkpoints:

| | Dice | HD95 | ASD |
|---|---|---|---|
| Retrained (3 seeds) | 87.07 ± 0.81 | 10.12 ± 1.20 | 2.73 ± 0.45 |
| Reported above | 86.95 ± 1.21 | 10.25 ± 2.61 | 2.87 ± 1.13 |

Every metric lands within 0.15 of the reported value. Per seed the retrain gives
87.98 (123), 86.80 (1337) and 86.42 (42), differing from the released checkpoints
by **+0.21, −0.73 and +0.86** — mixed signs and small magnitude, which is what
GPU-level nondeterminism looks like. A genuine recipe mismatch would push every
seed the same way, so that pattern is the thing to check if your own retrain
comes out low.

Expect run-to-run variation of about ±1 Dice on this split, and know where it
comes from: seed-averaged over the 20 test cases, 15 of 20 land within ±1 Dice of
the released model and the median difference is −0.06. Nearly all the spread is
carried by one hard case that some runs recover and others lose (up to 10 Dice on
that case alone). A single seed landing at 86.4 or 88.0 is normal; a mean far
outside 86–88 is not.

---

## The configuration

Every flag is already set in the scripts. Spelled out, the method is:

| | |
|---|---|
| Auxiliary head | boundary shell, **classification** (erosion kernel 3, XOR with the mask) |
| Backbone | dual-head V-Net, InstanceNorm |
| Semi-supervised | mean teacher, EMA decay 0.99, MSE consistency, ramp-up 40 |
| Pseudo-label gate | `--uncertainty_thresh 0.1` |
| Augmentation | copy-paste (labeled only), label smoothing 0.1 |
| Flip equivariance | axes 2 3 4, labeled stream, weight 0.5 |
| Loss weight | `--lambda_surface 1.0` |
| Schedule | 20k iterations, step LR (÷10 every 3333), no gradient clipping |

Two settings are dataset-specific and easy to get wrong:

- **Normalization.** LA and VerSe use InstanceNorm. ACDC uses **BatchNorm** —
  InstanceNorm destabilises 2 of 3 seeds there. `test.py` picks the right one per
  dataset automatically; a mismatch loads silently and produces garbage rather
  than raising.
- **Patch size** is fixed per dataset (LA 112×112×80, VerSe 128³, ACDC
  160×160×16) and must match between training and evaluation.

The auxiliary head is swappable via `--aux_head {boundary,sdf,bands,interface}`,
which is how the paper's classification-vs-regression ablation was run: `sdf` is
the SASSNet/DTC-style regression target under an otherwise identical framework.

## What is in here

```
train.py                 training entry point (all datasets)
test.py                  evaluate any checkpoint on a test set
networks/vnet_sdf.py     dual-head V-Net
dataloaders/             LA, VerSe, ACDC
utils/                   loss functions, consistency ramp-ups
shared/evaluation.py     the single evaluation path behind every reported number
shared/dataset_config.py dataset roots, patch sizes, class counts
scripts/                 turnkey train and test scripts
tools/summarize.py       aggregate per-seed CSVs into mean ± std
checkpoints/             trained weights, 3 seeds × 4 splits (+ SHA256SUMS)
LICENSE, THIRD_PARTY.md  MIT, plus attribution and the medpy/GPL note
```

ACDC is supported by `train.py` (`--dataset ACDC`, `--normalization batchnorm`)
but has no turnkey script here; the paper's ACDC rows use the same recipe with
7 or 14 labeled volumes and 20k iterations.

## Citation

If you use this code or the released checkpoints, please cite:

```bibtex
@article{kataria2025boundaryseg,
  title={BoundarySeg: An Embarrassingly Simple Method To Boost Medical Image Segmentation Performance for Low Data Regimes},
  author={Kataria, Tushar and Elhabian, Shireen Y},
  journal={arXiv preprint arXiv:2505.09829},
  year={2025}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Use it for anything, commercial included; the
only obligation is keeping the copyright notice.

One caveat worth reading before you build a product on it: `medpy`, used solely
to compute Dice/HD95/ASD in `shared/evaluation.py`, is **GPL-3.0**. That does
not affect the MIT grant over this code, but it can attach obligations to a
*distributed* combined work. It is one import in one file and nothing in the
model or training path depends on it. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for
that and for the code lineage this builds on.

## Notes

- CometML logging is **off** unless you set `COMET_API_KEY` or pass
  `--comet_key`. Nothing else phones home.
- `--ckpt_every` (default 5000) controls periodic `iter_*.pth` snapshots. These
  are large; set `--ckpt_every 0` to keep only the final checkpoint.
- Reproducibility across GPU models is bounded by cuDNN kernel selection; see
  the note under Results for the measured magnitude (~1e-4 per case).
