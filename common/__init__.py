"""
Shared datasets, transforms and training utilities for encoder-decoder models.
"""

from .data import (
    ISICBinarySegmentationDataset,
    build_transforms,
    collect_image_mask_pairs,
    create_dataloaders,
)
from .train_utils import (
    set_global_seeds,
    create_optimizer,
    create_loss,
    train_one_epoch,
    evaluate,
    run_training_loop,
)
from .runner import run_cli, build_parser

__all__ = [
    "ISICBinarySegmentationDataset",
    "build_transforms",
    "collect_image_mask_pairs",
    "create_dataloaders",
    "set_global_seeds",
    "create_optimizer",
    "create_loss",
    "train_one_epoch",
    "evaluate",
    "run_training_loop",
    "run_cli",
    "build_parser",
]

