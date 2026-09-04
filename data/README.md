# Datasets

Everything here is a **symlink**. No image data is committed; point these at
wherever the datasets already live on your machine:

```bash
ln -s /path/to/LA     data/LA
ln -s /path/to/VERSE  data/VERSE
ln -s /path/to/ACDC   data/ACDC     # optional, multi-class
```

The code resolves `<repo>/data/<DATASET>` by default, so nothing else needs
configuring. To keep the data elsewhere instead, set `BOUNDARYSEG_DATA_ROOT`
(and `BOUNDARYSEG_RESULTS_ROOT` for output), or pass `--la_root` / `--verse_root`
/ `--acdc_root` explicitly.

## Expected layout

**LA** — the standard 2018 Atrial Segmentation Challenge preprocessing used by
UA-MT / SASSNet / DTC. 80 training volumes, 20 test volumes; the image and its
label live together in one HDF5 file.

```
data/LA/
├── train.list                       # 80 case IDs, one per line
├── test.list                        # 20 case IDs
└── 2018LA_Seg_Training Set/
    └── <CASE_ID>/mri_norm2.h5       # datasets: 'image', 'label'
```

**VerSe** — 190 training volumes, 52 test volumes, resampled and cropped per
vertebra, one HDF5 file per case.

```
data/VERSE/
├── train.list
├── test.list
└── data/<CASE_ID>.h5                # datasets: 'image', 'label'
```

**ACDC** — nnU-Net / UNETR++ raw-data layout. `dataset.json`'s `validation`
entries are what evaluation scores (the test split ships no labels).

```
data/ACDC/
├── dataset.json
├── imagesTr/*.nii.gz
└── labelsTr/*.nii.gz
```

## Splits

The labeled/unlabeled split is positional, not random: with `--labelnum L` and
`--max_samples N`, cases `train.list[0:L]` are labeled and `train.list[L:N]` are
the unlabeled pool. So the split is fixed by the list file and identical across
seeds — only initialization and batch order change with `--seed`. Every baseline
in the paper was given the same split.
