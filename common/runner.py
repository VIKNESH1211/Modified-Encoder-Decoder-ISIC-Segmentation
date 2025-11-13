from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from . import (
    ISICBinarySegmentationDataset,
    build_transforms,
    collect_image_mask_pairs,
    create_dataloaders,
    create_loss,
    create_optimizer,
    evaluate,
    run_training_loop,
    set_global_seeds,
)
from .train_utils import create_model


def _parse_bool_flag(value: str) -> bool:
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}
    value_lower = value.lower()
    if value_lower in truthy:
        return True
    if value_lower in falsy:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value '{value}'.")


def build_parser(default_output_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a binary segmentation model using a specified encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-image-dir", required=True, help="Directory with training images.")
    parser.add_argument("--train-mask-dir", required=True, help="Directory with training masks.")
    parser.add_argument("--val-image-dir", help="Optional directory with validation images.")
    parser.add_argument("--val-mask-dir", help="Optional directory with validation masks.")
    parser.add_argument("--test-image-dir", help="Optional directory with test images for evaluation after training.")
    parser.add_argument("--test-mask-dir", help="Optional directory with test masks.")
    parser.add_argument("--mask-template", default="{stem}_Segmentation.png", help="Pattern to resolve mask filenames from image stems.")
    parser.add_argument("--image-extensions", nargs="+", default=["jpg", "jpeg", "png"], help="Image file extensions to include.")

    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size for both train and validation loaders.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes used by DataLoader.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize dimension for images and masks.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio when separate validation data is not provided.")
    parser.add_argument("--augment", type=_parse_bool_flag, default=True, help="Whether to apply simple data augmentation to the training set.")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam", help="Optimizer choice.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument("--loss", choices=["dice", "focal", "bce", "tversky"], default="dice", help="Loss function.")

    parser.add_argument("--encoder-weights", default="imagenet", help="Pretrained weights to use for the encoder. Use 'none' for random init.")
    parser.add_argument("--in-channels", type=int, default=3, help="Number of input image channels.")
    parser.add_argument("--classes", type=int, default=1, help="Number of output classes.")

    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold applied to probabilities for metric computation.")
    parser.add_argument("--monitor-metric", choices=["dice", "accuracy", "loss"], default="dice", help="Validation metric used for checkpointing.")
    parser.add_argument("--device", default="auto", help="Device to use: 'auto', 'cuda', 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--output-path", default=default_output_name, help="File path to store the best model weights.")
    parser.add_argument("--history-path", help="Optional JSON file to save the training history.")

    return parser


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _create_loaders_from_explicit_split(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    if not args.val_mask_dir:
        raise ValueError("When providing --val-image-dir you must also provide --val-mask-dir.")

    train_images, train_masks = collect_image_mask_pairs(
        args.train_image_dir,
        args.train_mask_dir,
        mask_template=args.mask_template,
        image_extensions=args.image_extensions,
    )
    val_images, val_masks = collect_image_mask_pairs(
        args.val_image_dir,
        args.val_mask_dir,
        mask_template=args.mask_template,
        image_extensions=args.image_extensions,
    )

    train_transform, val_transform = build_transforms(
        image_size=args.image_size,
        augment=args.augment,
    )

    train_dataset = ISICBinarySegmentationDataset(train_images, train_masks, transform=train_transform)
    val_dataset = ISICBinarySegmentationDataset(val_images, val_masks, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _create_test_loader(
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[DataLoader]:
    if not args.test_image_dir or not args.test_mask_dir:
        return None

    test_images, test_masks = collect_image_mask_pairs(
        args.test_image_dir,
        args.test_mask_dir,
        mask_template=args.mask_template,
        image_extensions=args.image_extensions,
    )
    _, val_transform = build_transforms(
        image_size=args.image_size,
        augment=False,
    )
    test_dataset = ISICBinarySegmentationDataset(
        test_images,
        test_masks,
        transform=val_transform,
    )
    return DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def run_cli(
    encoder_name: str,
    default_output_name: str,
    cli_args: Optional[Sequence[str]] = None,
) -> None:
    parser = build_parser(default_output_name)
    args = parser.parse_args(cli_args)

    set_global_seeds(args.seed)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights

    if args.val_image_dir:
        train_loader, val_loader = _create_loaders_from_explicit_split(args)
    else:
        train_loader, val_loader = create_dataloaders(
            args.train_image_dir,
            args.train_mask_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            seed=args.seed,
            mask_template=args.mask_template,
            image_extensions=args.image_extensions,
            augment=args.augment,
        )

    model = create_model(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=args.in_channels,
        classes=args.classes,
    ).to(device)

    loss_fn = create_loss(args.loss)
    optimizer = create_optimizer(
        model,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = run_training_loop(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        threshold=args.threshold,
        model_save_path=args.output_path,
        monitor_metric=args.monitor_metric,
    )

    if args.history_path:
        history_path = Path(args.history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_serialized = {
            "train": [metrics.__dict__ for metrics in history.train],
            "validation": [metrics.__dict__ for metrics in history.validation],
        }
        history_path.write_text(
            json.dumps(history_serialized, indent=2),
            encoding="utf-8",
        )

    test_loader = _create_test_loader(args, device)
    if test_loader is not None:
        print("\nEvaluating on the provided test set...")
        checkpoint_path = args.output_path
        if checkpoint_path and Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_metrics = evaluate(
            test_loader,
            model,
            loss_fn,
            device,
            threshold=args.threshold,
            desc="Test",
        )
        print(
            f"Test -> Loss: {test_metrics.loss:.4f}, "
            f"Acc: {test_metrics.accuracy*100:.2f}%, "
            f"Dice: {test_metrics.dice:.4f}"
        )


__all__ = ["run_cli", "build_parser"]

