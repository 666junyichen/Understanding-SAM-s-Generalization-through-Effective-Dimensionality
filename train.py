"""One-epoch training loops for SGD and SAM."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train a model for one epoch with a standard optimizer such as SGD."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += _count_correct(outputs, targets)
        total += batch_size

    return _format_metrics(running_loss, correct, total)


def train_one_epoch_sam(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Any,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train a model for one epoch with SAM.

    SAM uses two forward/backward passes per batch. The first pass perturbs the
    weights toward a nearby high-loss point; the second pass updates the model
    using gradients computed at that perturbed point.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        second_outputs = model(inputs)
        second_loss = criterion(second_outputs, targets)
        second_loss.backward()
        optimizer.second_step(zero_grad=True)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += _count_correct(outputs, targets)
        total += batch_size

    return _format_metrics(running_loss, correct, total)


def _count_correct(outputs: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).sum().item()


def _format_metrics(running_loss: float, correct: int, total: int) -> dict[str, float]:
    if total == 0:
        raise ValueError("Cannot compute training metrics for an empty dataloader.")

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }
