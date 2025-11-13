from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


class ISICBinarySegmentationDataset(Dataset):
    """
    PyTorch dataset for binary skin-lesion segmentation.

    Expects pre-built lists of image and mask paths of equal length.
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        mask_paths: Sequence[str],
        transform: Optional[Compose] = None,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError("The number of images and masks must match.")

        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.image_paths[index])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_paths[index]}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {self.mask_paths[index]}")

        mask = np.expand_dims(mask, axis=-1)
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.float()
        return image, mask


def _collect_images(
    image_dir: str,
    image_extensions: Iterable[str],
) -> List[Path]:
    paths: List[Path] = []
    for ext in image_extensions:
        paths.extend(
            Path(p)
            for p in glob.glob(os.path.join(image_dir, f"*.{ext}"))
        )
    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        exts = ", ".join(image_extensions)
        raise FileNotFoundError(
            f"No images found in '{image_dir}' with extensions: {exts}"
        )
    return unique_paths


def _build_mask_path(
    image_path: Path,
    mask_dir: Path,
    mask_template: str,
) -> Path:
    mask_name = mask_template.format(
        stem=image_path.stem,
        name=image_path.name,
    )
    return mask_dir / mask_name


def collect_image_mask_pairs(
    image_dir: str,
    mask_dir: str,
    *,
    mask_template: str = "{stem}_Segmentation.png",
    image_extensions: Sequence[str] = ("jpg", "jpeg", "png"),
) -> Tuple[List[str], List[str]]:
    """
    Assemble aligned image/mask path lists from the provided directories.
    """
    image_paths = _collect_images(image_dir, image_extensions)
    mask_dir_path = Path(mask_dir)

    mask_paths: List[Path] = []
    for image_path in image_paths:
        candidate = _build_mask_path(image_path, mask_dir_path, mask_template)
        if not candidate.exists():
            raise FileNotFoundError(
                f"Mask not found for image '{image_path.name}' at '{candidate}'."
            )
        mask_paths.append(candidate.resolve())

    return [str(p) for p in image_paths], [str(p) for p in mask_paths]


def build_transforms(
    image_size: int,
    augment: bool = True,
    horizontal_flip_prob: float = 0.5,
) -> Tuple[Compose, Compose]:
    """
    Build training and validation albumentations pipelines.
    """
    if augment:
        train_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=horizontal_flip_prob),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        train_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    val_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(
    image_dir: str,
    mask_dir: str,
    *,
    batch_size: int,
    image_size: int,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    shuffle: bool = True,
    mask_template: str = "{stem}_Segmentation.png",
    image_extensions: Sequence[str] = ("jpg", "jpeg", "png"),
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/validation dataloaders by splitting the dataset.
    """
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be between 0 and 1.")

    images, masks = collect_image_mask_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        mask_template=mask_template,
        image_extensions=image_extensions,
    )

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images,
        masks,
        test_size=val_split,
        random_state=seed,
        shuffle=shuffle,
    )

    train_transform, val_transform = build_transforms(
        image_size=image_size,
        augment=augment,
    )

    train_dataset = ISICBinarySegmentationDataset(
        train_imgs,
        train_masks,
        transform=train_transform,
    )
    val_dataset = ISICBinarySegmentationDataset(
        val_imgs,
        val_masks,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

