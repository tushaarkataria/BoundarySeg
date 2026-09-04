# Third-party code and dependencies

The code in this repository is MIT-licensed (see `LICENSE`). This file records
what it builds on, so downstream users can assess their own obligations.

## Code lineage

This work follows the SASSNet / UA-MT / DTC line of semi-supervised 3D medical
segmentation. Two files carry structure recognisable from that lineage:

| File | Origin |
|---|---|
| `networks/vnet_sdf.py` | V-Net implementation in the style used by SASSNet and DTC, extended here with a second output head (`out_conv`) for the boundary task |
| `dataloaders/la_heart.py` | LA HDF5 loader and `TwoStreamBatchSampler`, in the form used across UA-MT / SASSNet / DTC |

> **Confirm before publishing.** These upstream repositories are, to our
> knowledge, MIT-licensed, but that was not verified from the upstream
> `LICENSE` files while assembling this release. Check the current terms at
> the source repositories and add their copyright notices here if their license
> requires it. MIT, the most likely case, requires retaining the notice — which
> is why this section exists.

`utils/ramps.py` is a **clean-room** implementation of the Gaussian ramp-up
formula published in Laine & Aila, *Temporal Ensembling for Semi-Supervised
Learning* (ICLR 2017, arXiv:1610.02242), written from the paper's equation. It
deliberately contains no code from the Mean Teacher reference implementation,
whose `ramps.py` is licensed CC BY-NC 4.0 (NonCommercial) and would otherwise
have made this repository unusable commercially. The implementation is
numerically identical to the reference over the full input domain.

## Runtime dependencies

Everything in `requirements.txt` is permissively licensed **except one**:

| Package | License |
|---|---|
| `medpy` | **GPL-3.0-or-later** |
| torch, torchvision, numpy, scipy, scikit-image | BSD |
| nibabel, tensorboardX | MIT |
| kornia | Apache-2.0 |
| tqdm | MPL-2.0 AND MIT |
| h5py | BSD |

### The medpy caveat

`medpy` is GPL-3.0. It is a dependency, not vendored code — the only use is
`from medpy import metric` in `shared/evaluation.py`, which computes Dice,
Jaccard, HD95 and ASD. Nothing in the model, the training loop, or the
inference path touches it.

This does not affect the MIT grant over the code in this repository. It does
mean that anyone **distributing a combined work** that includes medpy should
consider whether GPL-3.0 obligations attach to that distribution. Two ways
around it, if that matters for your use:

- Use the repository for training and inference only, and compute metrics with
  your own implementation — `medpy` is imported in exactly one place.
- Replace the four metric calls with SciPy-based equivalents. Note that doing
  so will change the reported numbers slightly, so the exact-reproduction
  property documented in `README.md` no longer holds.

We are not lawyers and this is not legal advice.

## Datasets

No image data is distributed here. The LA (2018 Atrial Segmentation Challenge),
VerSe and ACDC datasets are obtained from their own providers under their own
terms, which are unaffected by this repository's license.
