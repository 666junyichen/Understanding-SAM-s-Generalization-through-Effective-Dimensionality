"""Evaluation helpers for ID and OOD performance."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model on one dataloader and return loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += batch_size

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def evaluate_all(
    model: nn.Module,
    id_loader: DataLoader,
    ood_loaders: dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one model on ID and all OOD dataloaders."""
    id_metrics = evaluate(model, id_loader, criterion, device)
    results = {
        "id_loss": id_metrics["loss"],
        "id_acc": id_metrics["accuracy"],
    }
    ood_accs = []

    for corruption_name, ood_loader in ood_loaders.items():
        ood_metrics = evaluate(model, ood_loader, criterion, device)
        results[f"ood_{corruption_name}_loss"] = ood_metrics["loss"]
        results[f"ood_{corruption_name}_acc"] = ood_metrics["accuracy"]
        ood_accs.append(ood_metrics["accuracy"])

    if not ood_accs:
        raise ValueError("At least one OOD dataloader is required.")

    avg_ood_acc = sum(ood_accs) / len(ood_accs)
    results["avg_ood_acc"] = avg_ood_acc
    results["id_ood_acc_drop"] = results["id_acc"] - avg_ood_acc

    return results
