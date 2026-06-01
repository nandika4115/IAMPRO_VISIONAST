"""
dataset.py
==========
PyTorch Dataset for VisionAST.
Handles both Dryad and ZIP dataset image/mask pairs with augmentation.
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List


# ── augmentation helpers ──────────────────────────────────────────────────────

def random_horizontal_flip(img, mask):
    if random.random() > 0.5:
        img  = cv2.flip(img,  1)
        mask = cv2.flip(mask, 1)
    return img, mask


def random_vertical_flip(img, mask):
    if random.random() > 0.5:
        img  = cv2.flip(img,  0)
        mask = cv2.flip(mask, 0)
    return img, mask


def random_rotate(img, mask, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img   = cv2.warpAffine(img,  M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    mask  = cv2.warpAffine(mask, M, (w, h),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REFLECT)
    return img, mask


def random_brightness_contrast(img, brightness=30, contrast=30):
    b = random.randint(-brightness, brightness)
    c = random.randint(-contrast,   contrast)
    img = img.astype(np.int32)
    img = img * (1 + c / 127.0) + b
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def elastic_transform(img, mask, alpha=80, sigma=8):
    """Light elastic deformation — excellent for biological image boundaries."""
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    img  = cv2.remap(img,  map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)
    mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_REFLECT)
    return img, mask


def augment(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img, mask = random_horizontal_flip(img, mask)
    img, mask = random_vertical_flip(img, mask)
    img, mask = random_rotate(img, mask)
    if random.random() > 0.5:
        img, mask = elastic_transform(img, mask)
    img = random_brightness_contrast(img)
    return img, mask


# ── dataset ───────────────────────────────────────────────────────────────────

class DiskDiffusionDataset(Dataset):
    """
    Pairs each plate image with its corresponding binary mask.

    images_dir : folder containing plate images  (jpg/png)
    masks_dir  : folder containing masks          (*_mask.png)
    img_size   : (H, W) to resize everything to
    augment    : apply augmentation (training only)
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        images_dir: str,
        masks_dir:  str,
        img_size:   Tuple[int, int] = (512, 512),
        do_augment: bool = False,
    ):
        self.images_dir  = Path(images_dir)
        self.masks_dir   = Path(masks_dir)
        self.img_size    = img_size
        self.do_augment  = do_augment

        self.pairs = self._find_pairs()
        if not self.pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found.\n"
                f"  images_dir = {images_dir}\n"
                f"  masks_dir  = {masks_dir}\n"
                f"  Make sure masks are named  <stem>_mask.png"
            )

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        pairs = []
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in self.IMG_EXTS:
                continue
            mask_path = self.masks_dir / (img_path.stem + "_mask.png")
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = cv2.imread(str(img_path))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # resize
        img  = cv2.resize(img,  self.img_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)

        # binarise mask
        mask = (mask > 127).astype(np.uint8)

        if self.do_augment:
            img, mask = augment(img, mask)

        # normalise image to [0,1] and convert to tensor
        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t


# ── split helper ──────────────────────────────────────────────────────────────

def make_splits(
    images_dirs: List[str],
    masks_dirs:  List[str],
    img_size:    Tuple[int, int] = (512, 512),
    val_frac:    float = 0.15,
    test_frac:   float = 0.10,
    seed:        int   = 42,
):
    """
    Merges multiple image/mask folder pairs, shuffles, and returns
    train / val / test Dataset objects.
    """
    from torch.utils.data import ConcatDataset, Subset

    # gather all pairs from all sources
    all_pairs = []
    for idir, mdir in zip(images_dirs, masks_dirs):
        ds = DiskDiffusionDataset(idir, mdir, img_size, do_augment=False)
        all_pairs.extend(ds.pairs)

    random.seed(seed)
    random.shuffle(all_pairs)
    n      = len(all_pairs)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_train= n - n_val - n_test

    def make_ds(pairs, augment):
        # tiny wrapper to create a dataset from an explicit list of pairs
        ds = _PairDataset(pairs, img_size, augment)
        return ds

    train_ds = make_ds(all_pairs[:n_train],          augment=True)
    val_ds   = make_ds(all_pairs[n_train:n_train+n_val], augment=False)
    test_ds  = make_ds(all_pairs[n_train+n_val:],    augment=False)

    print(f"Dataset split — train: {len(train_ds)}  "
          f"val: {len(val_ds)}  test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


class _PairDataset(Dataset):
    """Internal dataset built from an explicit list of (img_path, mask_path) pairs."""

    def __init__(self, pairs, img_size, do_augment):
        self.pairs      = pairs
        self.img_size   = img_size
        self.do_augment = do_augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = cv2.imread(str(img_path))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  self.img_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        if self.do_augment:
            img, mask = augment(img, mask)

        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t
