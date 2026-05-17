"""Generate figures and summary tables from experiment CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, RESULTS_DIR


def main() -> None:
    """Create report-ready plots and summary tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "metrics.csv"
    sgd_log_path = RESULTS_DIR / "training_log_sgd.csv"
    sam_log_path = RESULTS_DIR / "training_log_sam.csv"

    sgd_log, sam_log = load_training_logs(sgd_log_path, sam_log_path)
    metrics = load_metrics(metrics_path)

    plot_loss_curves(sgd_log, sam_log, FIGURES_DIR / "loss_plot.png")
    plot_accuracy_curves(sgd_log, sam_log, FIGURES_DIR / "accuracy_plot.png")
    plot_ood_bar_chart(metrics, FIGURES_DIR / "ood_bar_chart.png")
    plot_per_corruption_delta(metrics, FIGURES_DIR / "per_corruption_delta.png")
    plot_best_vs_final_id_acc(sgd_log, sam_log, FIGURES_DIR / "best_vs_final_id_acc.png")
    plot_checkpoint_comparison(metrics, FIGURES_DIR / "checkpoint_comparison.png")
    plot_generalization_gap(metrics, FIGURES_DIR / "generalization_gap.png")

    if _has_columns(sgd_log, ["avg_ood_acc"]) and _has_columns(sam_log, ["avg_ood_acc"]):
        plot_ood_trajectory(sgd_log, sam_log, FIGURES_DIR / "ood_trajectory.png")
    else:
        print("Skipped ood_trajectory.png because training logs do not contain OOD metrics yet.")

    if _has_columns(sgd_log, ["lr"]):
        plot_lr_schedule_curve(sgd_log, FIGURES_DIR / "lr_schedule_curve.png")
    else:
        print("Skipped lr_schedule_curve.png because training logs do not contain lr yet.")

    rho_sweep_path = RESULTS_DIR / "rho_sweep.csv"
    if rho_sweep_path.exists():
        plot_rho_sweep(pd.read_csv(rho_sweep_path), FIGURES_DIR / "rho_sweep.png")
    else:
        print("Skipped rho_sweep.png because results/rho_sweep.csv does not exist yet.")

    summary = create_summary_table(metrics_path, sgd_log_path, sam_log_path)
    summary.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    save_summary_table_png(summary, FIGURES_DIR / "summary_table.png")

    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved summary table to {RESULTS_DIR / 'summary_table.csv'}")


def load_training_logs(
    sgd_log_path: str | Path,
    sam_log_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SGD and SAM training logs."""
    return _read_csv(sgd_log_path), _read_csv(sam_log_path)


def load_metrics(metrics_path: str | Path) -> pd.DataFrame:
    """Load ID/OOD metrics for one or more checkpoint strategies."""
    return _read_csv(metrics_path)


def plot_loss_curves(
    sgd_log: pd.DataFrame,
    sam_log: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot SGD vs SAM training and ID loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sgd_log["epoch"], sgd_log["train_loss"], marker="o", label="SGD train loss")
    ax.plot(sgd_log["epoch"], sgd_log["id_loss"], marker="o", label="SGD ID loss")
    ax.plot(sam_log["epoch"], sam_log["train_loss"], marker="s", label="SAM-SGD train loss")
    ax.plot(sam_log["epoch"], sam_log["id_loss"], marker="s", label="SAM-SGD ID loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and ID Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_accuracy_curves(
    sgd_log: pd.DataFrame,
    sam_log: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot SGD vs SAM training and ID accuracy curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sgd_log["epoch"], sgd_log["train_acc"] * 100, marker="o", label="SGD train acc")
    ax.plot(sgd_log["epoch"], sgd_log["id_acc"] * 100, marker="o", label="SGD ID acc")
    ax.plot(sam_log["epoch"], sam_log["train_acc"] * 100, marker="s", label="SAM-SGD train acc")
    ax.plot(sam_log["epoch"], sam_log["id_acc"] * 100, marker="s", label="SAM-SGD ID acc")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training and ID Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_ood_bar_chart(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Plot ID/OOD robustness comparison for SGD and SAM."""
    metrics = _select_checkpoint_rows(metrics, preferred_checkpoint="best_ood")
    metrics = metrics.set_index("optimizer")
    categories = ["noise", "blur", "brightness", "avg_ood"]
    labels = ["Noise", "Blur", "Brightness", "Avg OOD"]

    sgd_values = [
        metrics.loc["sgd", "ood_noise_acc"],
        metrics.loc["sgd", "ood_blur_acc"],
        metrics.loc["sgd", "ood_brightness_acc"],
        metrics.loc["sgd", "avg_ood_acc"],
    ]
    sam_values = [
        metrics.loc["sam", "ood_noise_acc"],
        metrics.loc["sam", "ood_blur_acc"],
        metrics.loc["sam", "ood_brightness_acc"],
        metrics.loc["sam", "avg_ood_acc"],
    ]

    x_positions = range(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([x - width / 2 for x in x_positions], [v * 100 for v in sgd_values], width, label="SGD")
    ax.bar([x + width / 2 for x in x_positions], [v * 100 for v in sam_values], width, label="SAM-SGD")

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("OOD Accuracy Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_per_corruption_delta(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Plot SAM minus SGD accuracy for each OOD corruption."""
    metrics = _select_checkpoint_rows(metrics, preferred_checkpoint="best_ood").set_index("optimizer")
    categories = [
        ("ood_noise_acc", "Noise"),
        ("ood_blur_acc", "Blur"),
        ("ood_brightness_acc", "Brightness"),
        ("avg_ood_acc", "Avg OOD"),
    ]
    deltas = [(metrics.loc["sam", column] - metrics.loc["sgd", column]) * 100 for column, _ in categories]
    colors = ["#b84a4a" if value < 0 else "#2f7d5c" for value in deltas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([label for _, label in categories], deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("SAM-SGD Accuracy Difference (percentage points)")
    ax.set_title("Per-Corruption OOD Gain from SAM")
    ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, output_path)


def plot_best_vs_final_id_acc(
    sgd_log: pd.DataFrame,
    sam_log: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot best ID accuracy against final ID accuracy for each optimizer."""
    rows = [
        ("SGD", sgd_log),
        ("SAM-SGD", sam_log),
    ]
    labels = [label for label, _ in rows]
    best_values = [log["id_acc"].max() * 100 for _, log in rows]
    final_values = [log.iloc[-1]["id_acc"] * 100 for _, log in rows]

    x_positions = range(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([x - width / 2 for x in x_positions], best_values, width, label="Best ID")
    ax.bar([x + width / 2 for x in x_positions], final_values, width, label="Final ID")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("ID Accuracy (%)")
    ax.set_title("Best vs Final ID Accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_checkpoint_comparison(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Compare ID and average OOD accuracy across checkpoint strategies."""
    metrics = _ensure_checkpoint_metrics(metrics)
    metrics = _sort_checkpoint_metrics(metrics)
    labels = [
        f"{_format_optimizer(row['optimizer'])}\n{row['checkpoint_type']}"
        for _, row in metrics.iterrows()
    ]
    x_positions = range(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [x - width / 2 for x in x_positions],
        metrics["id_acc"] * 100,
        width,
        label="ID Acc",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        metrics["avg_ood_acc"] * 100,
        width,
        label="Avg OOD Acc",
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Checkpoint Strategy Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_ood_trajectory(
    sgd_log: pd.DataFrame,
    sam_log: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot average OOD accuracy over training epochs."""
    _require_columns(sgd_log, ["epoch", "avg_ood_acc"])
    _require_columns(sam_log, ["epoch", "avg_ood_acc"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sgd_log["epoch"], sgd_log["avg_ood_acc"] * 100, marker="o", label="SGD Avg OOD")
    ax.plot(sam_log["epoch"], sam_log["avg_ood_acc"] * 100, marker="s", label="SAM-SGD Avg OOD")

    for column, label in (
        ("ood_noise_acc", "Noise"),
        ("ood_blur_acc", "Blur"),
        ("ood_brightness_acc", "Brightness"),
    ):
        if column in sgd_log.columns and column in sam_log.columns:
            ax.plot(sgd_log["epoch"], sgd_log[column] * 100, alpha=0.25, linestyle="--")
            ax.plot(sam_log["epoch"], sam_log[column] * 100, alpha=0.25, linestyle=":")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("OOD Accuracy (%)")
    ax.set_title("OOD Robustness Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_generalization_gap(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Plot train-ID and ID-OOD generalization gaps."""
    metrics = _select_checkpoint_rows(metrics, preferred_checkpoint="best_ood")
    labels = [_format_optimizer(optimizer) for optimizer in metrics["optimizer"]]
    id_ood_gaps = metrics["id_ood_acc_drop"] * 100
    has_train_acc = "train_acc" in metrics.columns
    train_id_gaps = (metrics["train_acc"] - metrics["id_acc"]) * 100 if has_train_acc else None

    x_positions = range(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7, 5))
    if has_train_acc:
        ax.bar(
            [x - width / 2 for x in x_positions],
            train_id_gaps,
            width,
            label="Train-ID Gap",
        )
        ax.bar(
            [x + width / 2 for x in x_positions],
            id_ood_gaps,
            width,
            label="ID-OOD Gap",
        )
    else:
        ax.bar(list(x_positions), id_ood_gaps, width=0.5, label="ID-OOD Gap")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy Gap (percentage points)")
    ax.set_title("Generalization Gap")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def plot_lr_schedule_curve(training_log: pd.DataFrame, output_path: str | Path) -> None:
    """Plot the learning-rate schedule recorded in a training log."""
    _require_columns(training_log, ["epoch", "lr"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(training_log["epoch"], training_log["lr"], marker="o", color="#356a9a")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Learning-Rate Schedule")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, output_path)


def plot_rho_sweep(rho_sweep: pd.DataFrame, output_path: str | Path) -> None:
    """Plot validation performance across SAM rho values."""
    _require_columns(rho_sweep, ["rho", "best_val_acc"])
    rho_sweep = rho_sweep.sort_values("rho")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        rho_sweep["rho"].astype(str),
        rho_sweep["best_val_acc"] * 100,
        marker="o",
        linewidth=2,
        color="#2f7d5c",
    )

    best_index = rho_sweep["best_val_acc"].idxmax()
    best_row = rho_sweep.loc[best_index]
    ax.scatter(
        [str(best_row["rho"])],
        [best_row["best_val_acc"] * 100],
        color="#b84a4a",
        zorder=3,
        label=f"Best rho={best_row['rho']}",
    )

    ax.set_xlabel("SAM rho")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("SAM rho Validation Sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, output_path)


def create_summary_table(
    metrics_path: str | Path,
    sgd_log_path: str | Path,
    sam_log_path: str | Path,
) -> pd.DataFrame:
    """Create a compact report table from checkpoint metrics."""
    metrics = load_metrics(metrics_path)
    if "checkpoint_type" not in metrics.columns:
        sgd_final = _read_csv(sgd_log_path).iloc[-1]
        sam_final = _read_csv(sam_log_path).iloc[-1]
        old_metrics = metrics.set_index("optimizer")
        rows = [
            _summary_row("SGD", old_metrics.loc["sgd"], sgd_final),
            _summary_row("SAM-SGD", old_metrics.loc["sam"], sam_final),
        ]
        return pd.DataFrame(rows)

    rows = [_checkpoint_summary_row(row) for _, row in _sort_checkpoint_metrics(metrics).iterrows()]
    return pd.DataFrame(rows)


def save_summary_table_png(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Save a PNG image of the summary table for direct use in reports."""
    display_table = summary.copy()
    for column in display_table.columns:
        if column in {"Optimizer", "Checkpoint"}:
            continue
        if column == "Epoch":
            display_table[column] = display_table[column].map(lambda value: f"{int(value)}")
            continue
        if column == "Train Loss":
            display_table[column] = display_table[column].map(lambda value: f"{value:.4f}")
        else:
            display_table[column] = display_table[column].map(lambda value: f"{value * 100:.2f}%")

    fig_height = max(2.4, 0.55 * (len(display_table) + 1))
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)

    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    _save_figure(fig, output_path)


def _summary_row(label: str, metrics_row: pd.Series, final_log_row: pd.Series) -> dict[str, float | str]:
    return {
        "Optimizer": label,
        "Train Loss": final_log_row["train_loss"],
        "Train Acc": final_log_row["train_acc"],
        "ID Acc": metrics_row["id_acc"],
        "OOD Noise": metrics_row["ood_noise_acc"],
        "OOD Blur": metrics_row["ood_blur_acc"],
        "OOD Brightness": metrics_row["ood_brightness_acc"],
        "Avg OOD Acc": metrics_row["avg_ood_acc"],
        "ID-OOD Drop": metrics_row["id_ood_acc_drop"],
    }


def _checkpoint_summary_row(metrics_row: pd.Series) -> dict[str, float | str]:
    return {
        "Optimizer": _format_optimizer(metrics_row["optimizer"]),
        "Checkpoint": metrics_row["checkpoint_type"],
        "Epoch": metrics_row["epoch"],
        "Train Loss": metrics_row["train_loss"],
        "Train Acc": metrics_row["train_acc"],
        "ID Acc": metrics_row["id_acc"],
        "OOD Noise": metrics_row["ood_noise_acc"],
        "OOD Blur": metrics_row["ood_blur_acc"],
        "OOD Brightness": metrics_row["ood_brightness_acc"],
        "Avg OOD Acc": metrics_row["avg_ood_acc"],
        "ID-OOD Drop": metrics_row["id_ood_acc_drop"],
    }


def _select_checkpoint_rows(
    metrics: pd.DataFrame,
    preferred_checkpoint: str,
) -> pd.DataFrame:
    if "checkpoint_type" not in metrics.columns:
        return metrics

    available = set(metrics["checkpoint_type"])
    checkpoint_type = preferred_checkpoint if preferred_checkpoint in available else "last"
    return metrics[metrics["checkpoint_type"] == checkpoint_type]


def _ensure_checkpoint_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if "checkpoint_type" in metrics.columns:
        return metrics
    return metrics.assign(checkpoint_type="last")


def _has_columns(dataframe: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in dataframe.columns for column in columns)


def _require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _format_optimizer(optimizer: str) -> str:
    if optimizer == "sam":
        return "SAM-SGD"
    return optimizer.upper()


def _sort_checkpoint_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    optimizer_order = {"sgd": 0, "sam": 1}
    checkpoint_order = {"last": 0, "best_id": 1, "best_ood": 2}
    return (
        metrics.assign(
            _optimizer_order=metrics["optimizer"].map(optimizer_order).fillna(99),
            _checkpoint_order=metrics["checkpoint_type"].map(checkpoint_order).fillna(99),
        )
        .sort_values(["_optimizer_order", "_checkpoint_order", "optimizer", "checkpoint_type"])
        .drop(columns=["_optimizer_order", "_checkpoint_order"])
    )


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run run_experiment.py first.")
    return pd.read_csv(csv_path)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
