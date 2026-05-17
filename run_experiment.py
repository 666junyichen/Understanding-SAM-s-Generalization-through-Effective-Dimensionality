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
    LR_SCHEDULER,
    MODEL_NAME,
    MOMENTUM,
    RESULTS_DIR,
    SAM_RHO,
    SEED,
    WEIGHT_DECAY,
)
from data import get_id_test_loader, get_ood_loaders, get_train_loader
from evaluate import evaluate_all
from models import get_model
from sam import SAM
from train import train_one_epoch_sam, train_one_epoch_sgd
from utils import create_dirs, get_device, save_checkpoint, set_seed


TRAINING_LOG_FIELDNAMES = [
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
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

METRICS_FIELDNAMES = [
    "optimizer",
    "checkpoint_type",
    *TRAINING_LOG_FIELDNAMES,
]


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
        training_log, checkpoint_metrics = run_single_experiment(
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
            TRAINING_LOG_FIELDNAMES,
        )
        all_metrics.extend(checkpoint_metrics)

    save_csv(RESULTS_DIR / "metrics.csv", all_metrics, _metric_fieldnames(all_metrics))
    print(f"Saved checkpoint metrics to {RESULTS_DIR / 'metrics.csv'}")


def run_single_experiment(
    optimizer_name: str,
    initial_state: dict[str, torch.Tensor],
    train_loader,
    id_loader,
    ood_loaders,
    device: torch.device,
) -> tuple[list[dict[str, float]], list[dict[str, float | str]]]:
    """Train and evaluate one optimizer from a shared initialization."""
    model = get_model(MODEL_NAME).to(device)
    model.load_state_dict(initial_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model)
    scheduler = build_scheduler(optimizer)

    training_log = []
    checkpoint_metrics = {}
    best_id_acc = float("-inf")
    best_avg_ood_acc = float("-inf")
    train_one_epoch = train_one_epoch_sam if optimizer_name == "sam" else train_one_epoch_sgd

    for epoch in range(1, EPOCHS + 1):
        current_lr = get_current_lr(optimizer)
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_metrics = evaluate_all(model, id_loader, ood_loaders, criterion, device)
        epoch_row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            **eval_metrics,
        }
        training_log.append(epoch_row)

        if epoch_row["id_acc"] > best_id_acc:
            best_id_acc = epoch_row["id_acc"]
            checkpoint_metrics["best_id"] = dict(epoch_row)
            _save_experiment_checkpoint(
                optimizer_name, "best_id", epoch_row, model, optimizer, scheduler
            )

        if epoch_row["avg_ood_acc"] > best_avg_ood_acc:
            best_avg_ood_acc = epoch_row["avg_ood_acc"]
            checkpoint_metrics["best_ood"] = dict(epoch_row)
            _save_experiment_checkpoint(
                optimizer_name, "best_ood", epoch_row, model, optimizer, scheduler
            )

        print(
            f"[{optimizer_name.upper()}] "
            f"Epoch {epoch}/{EPOCHS} "
            f"lr={current_lr:.6f} "
            f"train_acc={epoch_row['train_acc']:.4f} "
            f"id_acc={epoch_row['id_acc']:.4f} "
            f"avg_ood_acc={epoch_row['avg_ood_acc']:.4f}"
        )
        scheduler.step()

    checkpoint_metrics["last"] = dict(training_log[-1])
    _save_experiment_checkpoint(optimizer_name, "last", training_log[-1], model, optimizer, scheduler)

    metrics_rows = []
    for checkpoint_type in ("last", "best_id", "best_ood"):
        metrics_rows.append(
            {
                "optimizer": optimizer_name,
                "checkpoint_type": checkpoint_type,
                **checkpoint_metrics[checkpoint_type],
            }
        )

    return training_log, metrics_rows


def _save_experiment_checkpoint(
    optimizer_name: str,
    checkpoint_type: str,
    metrics: dict[str, float],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    checkpoint_path = (
        CHECKPOINT_DIR / f"{optimizer_name}_{MODEL_NAME.lower()}_cifar10_{checkpoint_type}.pt"
    )
    save_checkpoint(
        {
            "optimizer_name": optimizer_name,
            "checkpoint_type": checkpoint_type,
            "model_name": MODEL_NAME,
            "epoch": metrics["epoch"],
            "lr_scheduler": LR_SCHEDULER,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print(f"Saved {optimizer_name.upper()} {checkpoint_type} checkpoint to {checkpoint_path}")


def build_optimizer(
    optimizer_name: str,
    model: nn.Module,
    sam_rho: float = SAM_RHO,
) -> torch.optim.Optimizer:
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
            rho=sam_rho,
        )

    raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    """Create the learning-rate scheduler for the shared training protocol."""
    if LR_SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if LR_SCHEDULER in {"none", None}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    raise ValueError(f"Unknown LR scheduler: {LR_SCHEDULER}")


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate from the first optimizer parameter group."""
    return float(optimizer.param_groups[0]["lr"])


def save_csv(path: str | Path, rows: list[dict[str, float | str]], fieldnames: list[str]) -> None:
    """Save rows to a CSV file with a stable column order."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_fieldnames(rows: list[dict[str, float | str]]) -> list[str]:
    if not rows:
        raise ValueError("Cannot infer metric columns from an empty result list.")

    extra = sorted({key for row in rows for key in row if key not in METRICS_FIELDNAMES})
    return [key for key in METRICS_FIELDNAMES if key in rows[0]] + extra


if __name__ == "__main__":
    main()
