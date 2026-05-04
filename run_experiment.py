"""Main entry point for SGD vs SAM-SGD CIFAR-10 experiments."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn

from config import (
    CHECKPOINT_DIR,
    DEBUG_SUBSET_SIZE,
    EPOCHS,
    LR,
    MODEL_NAME,
    MOMENTUM,
    RESULTS_DIR,
    SAM_RHO,
    SEED,
    WEIGHT_DECAY,
)
from data import get_id_test_loader, get_ood_loaders, get_train_loader
from evaluate import evaluate, evaluate_all
from models import get_model
from sam import SAM
from train import train_one_epoch_sam, train_one_epoch_sgd
from utils import create_dirs, get_device, save_checkpoint, set_seed


def main() -> None:
    """Run SGD and SAM-SGD experiments and save checkpoints and metrics."""
    set_seed(SEED)
    create_dirs()
    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_train_loader(subset_size=DEBUG_SUBSET_SIZE)
    id_loader = get_id_test_loader(subset_size=DEBUG_SUBSET_SIZE)
    ood_loaders = get_ood_loaders(subset_size=DEBUG_SUBSET_SIZE)

    initial_model = get_model(MODEL_NAME)
    initial_state = copy.deepcopy(initial_model.state_dict())

    all_metrics = []
    for optimizer_name in ("sgd", "sam"):
        training_log, final_metrics = run_single_experiment(
            optimizer_name=optimizer_name,
            initial_state=initial_state,
            train_loader=train_loader,
            id_loader=id_loader,
            ood_loaders=ood_loaders,
            device=device,
        )
        save_csv(
            RESULTS_DIR / f"training_log_{optimizer_name}.csv",
            training_log,
            ["epoch", "train_loss", "train_acc", "id_loss", "id_acc"],
        )
        all_metrics.append({"optimizer": optimizer_name, **final_metrics})

    save_csv(RESULTS_DIR / "metrics.csv", all_metrics, _metric_fieldnames(all_metrics))
    print(f"Saved final metrics to {RESULTS_DIR / 'metrics.csv'}")


def run_single_experiment(
    optimizer_name: str,
    initial_state: dict[str, torch.Tensor],
    train_loader,
    id_loader,
    ood_loaders,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Train and evaluate one optimizer from a shared initialization."""
    model = get_model(MODEL_NAME).to(device)
    model.load_state_dict(initial_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model)

    training_log = []
    train_one_epoch = train_one_epoch_sam if optimizer_name == "sam" else train_one_epoch_sgd

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        id_metrics = evaluate(model, id_loader, criterion, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "id_loss": id_metrics["loss"],
            "id_acc": id_metrics["accuracy"],
        }
        training_log.append(epoch_row)
        print(
            f"[{optimizer_name.upper()}] "
            f"Epoch {epoch}/{EPOCHS} "
            f"train_acc={epoch_row['train_acc']:.4f} "
            f"id_acc={epoch_row['id_acc']:.4f}"
        )

    final_metrics = evaluate_all(model, id_loader, ood_loaders, criterion, device)
    checkpoint_path = CHECKPOINT_DIR / f"{optimizer_name}_{MODEL_NAME.lower()}_cifar10.pt"
    save_checkpoint(
        {
            "optimizer_name": optimizer_name,
            "model_name": MODEL_NAME,
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_metrics": final_metrics,
        },
        checkpoint_path,
    )
    print(f"Saved {optimizer_name.upper()} checkpoint to {checkpoint_path}")

    return training_log, final_metrics


def build_optimizer(optimizer_name: str, model: nn.Module) -> torch.optim.Optimizer:
    """Create the optimizer used by one experimental condition."""
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
    if optimizer_name == "sam":
        return SAM(
            model.parameters(),
            base_optimizer=torch.optim.SGD,
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            rho=SAM_RHO,
        )

    raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def save_csv(path: str | Path, rows: list[dict[str, float]], fieldnames: list[str]) -> None:
    """Save rows to a CSV file with a stable column order."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_fieldnames(rows: list[dict[str, float]]) -> list[str]:
    if not rows:
        raise ValueError("Cannot infer metric columns from an empty result list.")

    preferred = [
        "optimizer",
        "id_loss",
        "id_acc",
        "ood_noise_loss",
        "ood_noise_acc",
        "ood_blur_loss",
        "ood_blur_acc",
        "ood_brightness_loss",
        "ood_brightness_acc",
        "avg_ood_acc",
        "id_ood_acc_drop",
    ]
    extra = sorted({key for row in rows for key in row if key not in preferred})
    return [key for key in preferred if key in rows[0]] + extra


if __name__ == "__main__":
    main()
