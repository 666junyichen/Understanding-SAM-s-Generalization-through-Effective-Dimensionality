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
    """Load final ID/OOD metrics."""
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


def create_summary_table(
    metrics_path: str | Path,
    sgd_log_path: str | Path,
    sam_log_path: str | Path,
) -> pd.DataFrame:
    """Create a compact report table from metrics and final epoch logs."""
    metrics = load_metrics(metrics_path).set_index("optimizer")
    sgd_final = _read_csv(sgd_log_path).iloc[-1]
    sam_final = _read_csv(sam_log_path).iloc[-1]

    rows = [
        _summary_row("SGD", metrics.loc["sgd"], sgd_final),
        _summary_row("SAM-SGD", metrics.loc["sam"], sam_final),
    ]
    return pd.DataFrame(rows)


def save_summary_table_png(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Save a PNG image of the summary table for direct use in reports."""
    display_table = summary.copy()
    for column in display_table.columns:
        if column == "Optimizer":
            continue
        if column == "Train Loss":
            display_table[column] = display_table[column].map(lambda value: f"{value:.4f}")
        else:
            display_table[column] = display_table[column].map(lambda value: f"{value * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 2.4))
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
