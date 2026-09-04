"""
ACDC cardiac cine-MRI dataset loader.

Structure on disk (nnUNet/UNETR++ raw-data format):
  <root>/imagesTr/patientNNN_frameFF_0000.nii.gz   (MRI, float)
  <root>/labelsTr/patientNNN_frameFF_gt.nii.gz     (int, 0-3)
  <root>/imagesTs                                  (test split, no labels)
  <root>/dataset.json                              (authoritative file lists)

Labels (4 classes including background):
  0  background
  1  right ventricular cavity (RV)
  2  myocardium
  3  left ventricular cavity (LV)

NOTE: the test split in dataset.json has no labels — use 'val' for evaluation.

Native volumes are anisotropic short-axis MRI, ~10 slices deep with variable
in-plane size (~200-256 px). Use patch_size (160, 160, 16) -- not AMOS's
(96, 96, 96), or most of the patch will be zero-padding -- and not less than
16 in Z, which is the minimum that survives VNet's 4 stride-2 downsamples.
"""

import os
import json
import numpy as np
from torch.utils.data import Dataset

NUM_CLASSES = 4  # 0=background, 1=RV, 2=myocardium, 3=LV

CLASS_NAMES = ['background', 'RV', 'myocardium', 'LV']


def normalize_intensity(image):
    """Per-volume percentile clip + min-max scale to [0, 1].

    Raw MRI intensity has no fixed unit scale like CT's HU window, so the
    clip range is computed per-volume rather than using a fixed constant.
    """
    p_lo, p_hi = np.percentile(image, (0.5, 99.5))
    image = np.clip(image, p_lo, p_hi)
    image = (image - p_lo) / (p_hi - p_lo + 1e-8)
    return image.astype(np.float32)


class ACDC(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root      = root
        self.transform = transform

        with open(os.path.join(root, 'dataset.json')) as f:
            meta = json.load(f)

        if split == 'train':
            entries = meta['training']
        elif split == 'val':
            entries = meta['validation']
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.image_paths = [os.path.join(root, e['image'].lstrip('./')) for e in entries]
        self.label_paths = [os.path.join(root, e['label'].lstrip('./')) for e in entries]
        print(f'ACDC {split}: {len(self.image_paths)} cases, {NUM_CLASSES} classes')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import nibabel as nib
        image = nib.load(self.image_paths[idx]).get_fdata(dtype=np.float32)
        label = nib.load(self.label_paths[idx]).get_fdata().astype(np.int64)

        image = normalize_intensity(image)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
