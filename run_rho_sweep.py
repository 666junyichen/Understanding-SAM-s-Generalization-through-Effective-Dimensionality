"""Sweep SAM rho values on a validation split for Part A."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from config import DEBUG_SUBSET_SIZE, EPOCHS, RESULTS_DIR, SAM_RHO_SWEEP_VALUES, SEED
from data import get_train_val_loaders
from evaluate import evaluate
from models import get_model
from run_experiment import build_optimizer, build_scheduler, get_current_lr, save_csv
from train import train_one_epoch_sam
from utils import create_dirs, get_device, set_seed


RHO_SWEEP_FIELDNAMES = [
    "rho",
    "best_epoch",
    "best_val_loss",
    "best_val_acc",
    "final_val_loss",
    "final_val_acc",
]


def main() -> None:
    """Run a validation sweep over candidate SAM rho values."""
    set_seed(SEED)
    create_dirs()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = get_train_val_loaders(subset_size=DEBUG_SUBSET_SIZE)
    initial_model = get_model()
    initial_state = copy.deepcopy(initial_model.state_dict())

    rows = []
    for rho in SAM_RHO_SWEEP_VALUES:
        row = run_single_rho(
            rho=float(rho),
            initial_state=initial_state,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        rows.append(row)

    output_path = RESULTS_DIR / "rho_sweep.csv"
    save_csv(output_path, rows, RHO_SWEEP_FIELDNAMES)
    print(f"Saved rho sweep results to {output_path}")


def run_single_rho(
    rho: float,
    initial_state: dict[str, torch.Tensor],
    train_loader,
    val_loader,
    device: torch.device,
) -> dict[str, float | int]:
    """Train SAM for one rho value and return best validation metrics."""
    model = get_model().to(device)
    model.load_state_dict(initial_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer("sam", model, sam_rho=rho)
    scheduler = build_scheduler(optimizer)

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    final_val_loss = float("nan")
    final_val_acc = float("nan")

    for epoch in range(1, EPOCHS + 1):
        current_lr = get_current_lr(optimizer)
        train_metrics = train_one_epoch_sam(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        final_val_loss = val_metrics["loss"]
        final_val_acc = val_metrics["accuracy"]

        if final_val_acc > best_val_acc:
            best_epoch = epoch
            best_val_loss = final_val_loss
            best_val_acc = final_val_acc

        print(
            f"[SAM rho={rho:.3g}] "
            f"Epoch {epoch}/{EPOCHS} "
            f"lr={current_lr:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={final_val_acc:.4f}"
        )
        scheduler.step()

    return {
        "rho": rho,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
    }


if __name__ == "__main__":
    main()
