"""
VERSE vertebra segmentation dataset loader.

Each H5 file is one pre-cropped vertebra patch (128×128×128) extracted from
a full spine CT.  The filename encodes subject and vertebra level:
    sub-verse{ID}_seg-vert_msk_{VERTEBRA_LABEL}.h5

H5 keys:
    image  (128,128,128) float32  — CT intensity, already normalised to [0,1]
    label  (128,128,128) float64  — binary mask (0=background, 1=vertebra)
    SDF    (128,128,128) float64  — pre-computed signed distance map

Binary segmentation (num_classes=2).  The SDF key means --method surface_sdf
works without any extra preprocessing.

Split files (train.list / test.list) are in the dataset root alongside data/.
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class VERSE(Dataset):
    """
    Loads VERSE vertebra patches from H5 files listed in train.list / test.list.
    Returns {'image', 'label'} or {'image', 'label', 'sdf'} depending on
    whether with_sdf=True.
    """

    def __init__(self, base_dir, split='train', transform=None, with_sdf=False):
        self.base_dir = base_dir
        self.transform = transform
        self.with_sdf  = with_sdf

        list_file = os.path.join(base_dir, f'{split}.list')
        with open(list_file) as f:
            self.samples = [l.strip() for l in f if l.strip()]

        print(f'VERSE {split}: {len(self.samples)} vertebra patches')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = os.path.join(self.base_dir, 'data', self.samples[idx] + '.h5')
        with h5py.File(path, 'r') as f:
            image = f['image'][:]                        # float32, [0,1]
            label = f['label'][:].astype(np.uint8)      # binary {0,1}
            sdf   = f['SDF'][:].astype(np.float32) if self.with_sdf else None

        sample = {'image': image, 'label': label}
        if self.with_sdf:
            sample['sdf'] = sdf

        if self.transform:
            sample = self.transform(sample)
        return sample
