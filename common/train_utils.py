from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import segmentation_models_pytorch as smp


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    dice: float


@dataclass
class TrainingHistory:
    train: List[EpochMetrics] = field(default_factory=list)
    validation: List[EpochMetrics] = field(default_factory=list)

    def update(self, train_metrics: EpochMetrics, val_metrics: EpochMetrics) -> None:
        self.train.append(train_metrics)
        self.validation.append(val_metrics)


def set_global_seeds(seed: int) -> None:
    """
    Ensure deterministic behavior across Python, NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute pixel accuracy and Dice score for binary predictions.
    """
    probs = torch.sigmoid(predictions)
    preds = (probs > threshold).float()

    correct = (preds == targets).float().sum().item()
    total = float(targets.numel())
    accuracy = correct / total

    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)

    return accuracy, dice


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    threshold: float = 0.5,
    desc: str = "Training",
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_pixels = 0.0
    total_intersection = 0.0
    total_union = 0.0

    progress_bar = tqdm(dataloader, desc=desc, leave=False)
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            total_correct += (preds == masks).float().sum().item()
            total_pixels += float(masks.numel())

            intersection = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item()
            total_intersection += intersection
            total_union += union

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_pixels
    dice = (2.0 * total_intersection + 1e-7) / (total_union + 1e-7)
    return EpochMetrics(loss=avg_loss, accuracy=accuracy, dice=dice)


@torch.no_grad()
def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    desc: str = "Validation",
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_pixels = 0.0
    total_intersection = 0.0
    total_union = 0.0

    progress_bar = tqdm(dataloader, desc=desc, leave=False)
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        total_correct += (preds == masks).float().sum().item()
        total_pixels += float(masks.numel())

        intersection = (preds * masks).sum().item()
        union = preds.sum().item() + masks.sum().item()
        total_intersection += intersection
        total_union += union

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_pixels
    dice = (2.0 * total_intersection + 1e-7) / (total_union + 1e-7)
    return EpochMetrics(loss=avg_loss, accuracy=accuracy, dice=dice)


def create_model(
    encoder_name: str,
    *,
    encoder_weights: Optional[str],
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


def create_loss(
    loss_name: str = "dice",
    *,
    from_logits: bool = True,
    beta: float = 0.5,
) -> nn.Module:
    if loss_name.lower() == "dice":
        return smp.losses.DiceLoss(mode="binary", from_logits=from_logits)
    if loss_name.lower() == "focal":
        return smp.losses.FocalLoss(mode="binary")
    if loss_name.lower() == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name.lower() == "tversky":
        return smp.losses.TverskyLoss(mode="binary", log_loss=False, alpha=beta, beta=beta)
    raise ValueError(f"Unsupported loss function '{loss_name}'.")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> optim.Optimizer:
    params = model.parameters()
    name = optimizer_name.lower()

    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    threshold: float = 0.5,
    model_save_path: Optional[str] = None,
    monitor_metric: str = "dice",
) -> TrainingHistory:
    history = TrainingHistory()
    best_metric = -math.inf

    model.to(device)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_metrics = train_one_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
            threshold=threshold,
            desc="Train",
        )
        val_metrics = evaluate(
            val_loader,
            model,
            loss_fn,
            device,
            threshold=threshold,
            desc="Val",
        )

        history.update(train_metrics, val_metrics)
        print(
            f"Train -> Loss: {train_metrics.loss:.4f}, "
            f"Acc: {train_metrics.accuracy*100:.2f}%, "
            f"Dice: {train_metrics.dice:.4f}"
        )
        print(
            f"Val   -> Loss: {val_metrics.loss:.4f}, "
            f"Acc: {val_metrics.accuracy*100:.2f}%, "
            f"Dice: {val_metrics.dice:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_metrics.loss)

        current_metric = getattr(val_metrics, monitor_metric, None)
        if current_metric is None:
            raise AttributeError(
                f"Validation metrics do not contain '{monitor_metric}'. "
                f"Available: {val_metrics.__dict__.keys()}"
            )

        if model_save_path and current_metric > best_metric:
            best_metric = current_metric
            directory = os.path.dirname(model_save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved new best model to '{model_save_path}' (best {monitor_metric}: {best_metric:.4f})")

    return history

