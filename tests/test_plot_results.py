import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class PlotResultsTests(unittest.TestCase):
    def test_create_summary_table_combines_metrics_and_final_training_rows(self):
        from plot_results import create_summary_table

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "metrics.csv"
            sgd_log_path = tmp_path / "training_log_sgd.csv"
            sam_log_path = tmp_path / "training_log_sam.csv"

            pd.DataFrame(
                [
                    {
                        "optimizer": "sgd",
                        "id_acc": 0.8,
                        "ood_noise_acc": 0.6,
                        "ood_blur_acc": 0.7,
                        "ood_brightness_acc": 0.75,
                        "avg_ood_acc": 0.6833333333,
                        "id_ood_acc_drop": 0.1166666667,
                    },
                    {
                        "optimizer": "sam",
                        "id_acc": 0.85,
                        "ood_noise_acc": 0.7,
                        "ood_blur_acc": 0.76,
                        "ood_brightness_acc": 0.8,
                        "avg_ood_acc": 0.7533333333,
                        "id_ood_acc_drop": 0.0966666667,
                    },
                ]
            ).to_csv(metrics_path, index=False)
            pd.DataFrame([{"epoch": 1, "train_loss": 1.2, "train_acc": 0.55}]).to_csv(
                sgd_log_path, index=False
            )
            pd.DataFrame([{"epoch": 1, "train_loss": 1.1, "train_acc": 0.58}]).to_csv(
                sam_log_path, index=False
            )

            summary = create_summary_table(metrics_path, sgd_log_path, sam_log_path)

        self.assertEqual(
            list(summary.columns),
            [
                "Optimizer",
                "Train Loss",
                "Train Acc",
                "ID Acc",
                "OOD Noise",
                "OOD Blur",
                "OOD Brightness",
                "Avg OOD Acc",
                "ID-OOD Drop",
            ],
        )
        self.assertEqual(summary["Optimizer"].tolist(), ["SGD", "SAM-SGD"])

    def test_create_summary_table_expands_checkpoint_rows(self):
        from plot_results import create_summary_table

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "metrics.csv"
            sgd_log_path = tmp_path / "training_log_sgd.csv"
            sam_log_path = tmp_path / "training_log_sam.csv"

            base_row = {
                "epoch": 1,
                "train_loss": 1.0,
                "train_acc": 0.5,
                "id_loss": 0.8,
                "id_acc": 0.7,
                "ood_noise_loss": 1.1,
                "ood_noise_acc": 0.4,
                "ood_blur_loss": 0.9,
                "ood_blur_acc": 0.6,
                "ood_brightness_loss": 0.7,
                "ood_brightness_acc": 0.65,
                "avg_ood_acc": 0.55,
                "id_ood_acc_drop": 0.15,
            }
            rows = [
                {"optimizer": optimizer, "checkpoint_type": checkpoint, **base_row}
                for optimizer in ("sgd", "sam")
                for checkpoint in ("last", "best_id", "best_ood")
            ]

            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            pd.DataFrame([base_row]).to_csv(sgd_log_path, index=False)
            pd.DataFrame([base_row]).to_csv(sam_log_path, index=False)

            summary = create_summary_table(metrics_path, sgd_log_path, sam_log_path)

        self.assertEqual(len(summary), 6)
        self.assertEqual(
            list(summary.columns),
            [
                "Optimizer",
                "Checkpoint",
                "Epoch",
                "Train Loss",
                "Train Acc",
                "ID Acc",
                "OOD Noise",
                "OOD Blur",
                "OOD Brightness",
                "Avg OOD Acc",
                "ID-OOD Drop",
            ],
        )
        self.assertEqual(
            summary[["Optimizer", "Checkpoint"]].values.tolist(),
            [
                ["SGD", "last"],
                ["SGD", "best_id"],
                ["SGD", "best_ood"],
                ["SAM-SGD", "last"],
                ["SAM-SGD", "best_id"],
                ["SAM-SGD", "best_ood"],
            ],
        )

    def test_new_part_a_plots_create_png_files(self):
        from plot_results import (
            plot_best_vs_final_id_acc,
            plot_checkpoint_comparison,
            plot_generalization_gap,
            plot_ood_trajectory,
            plot_per_corruption_delta,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics = pd.DataFrame(
                [
                    _metrics_row("sgd", "last", 50, 0.77, 0.20, 0.58, 0.67, 0.48),
                    _metrics_row("sgd", "best_id", 40, 0.85, 0.21, 0.59, 0.68, 0.49),
                    _metrics_row("sgd", "best_ood", 42, 0.84, 0.22, 0.60, 0.69, 0.50),
                    _metrics_row("sam", "last", 50, 0.84, 0.17, 0.61, 0.80, 0.53),
                    _metrics_row("sam", "best_id", 40, 0.88, 0.18, 0.62, 0.81, 0.54),
                    _metrics_row("sam", "best_ood", 43, 0.87, 0.19, 0.63, 0.82, 0.55),
                ]
            )
            sgd_log = pd.DataFrame(
                [
                    {"epoch": 1, "train_acc": 0.5, "id_acc": 0.6, "avg_ood_acc": 0.4},
                    {"epoch": 2, "train_acc": 0.7, "id_acc": 0.8, "avg_ood_acc": 0.5},
                ]
            )
            sam_log = pd.DataFrame(
                [
                    {"epoch": 1, "train_acc": 0.55, "id_acc": 0.65, "avg_ood_acc": 0.45},
                    {"epoch": 2, "train_acc": 0.75, "id_acc": 0.85, "avg_ood_acc": 0.55},
                ]
            )

            outputs = {
                "delta": tmp_path / "per_corruption_delta.png",
                "best_final": tmp_path / "best_vs_final_id_acc.png",
                "checkpoint": tmp_path / "checkpoint_comparison.png",
                "trajectory": tmp_path / "ood_trajectory.png",
                "gap": tmp_path / "generalization_gap.png",
            }

            plot_per_corruption_delta(metrics, outputs["delta"])
            plot_best_vs_final_id_acc(sgd_log, sam_log, outputs["best_final"])
            plot_checkpoint_comparison(metrics, outputs["checkpoint"])
            plot_ood_trajectory(sgd_log, sam_log, outputs["trajectory"])
            plot_generalization_gap(metrics, outputs["gap"])

            for output_path in outputs.values():
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)

    def test_lr_schedule_plot_creates_png_file(self):
        from plot_results import plot_lr_schedule_curve

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_log = pd.DataFrame(
                [
                    {"epoch": 1, "lr": 0.1},
                    {"epoch": 2, "lr": 0.05},
                    {"epoch": 3, "lr": 0.0},
                ]
            )

            lr_output = tmp_path / "lr_schedule_curve.png"

            plot_lr_schedule_curve(training_log, lr_output)

            self.assertTrue(lr_output.exists())
            self.assertGreater(lr_output.stat().st_size, 0)


def _metrics_row(
    optimizer,
    checkpoint_type,
    epoch,
    id_acc,
    noise_acc,
    blur_acc,
    brightness_acc,
    avg_ood_acc,
):
    return {
        "optimizer": optimizer,
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "train_loss": 1.0,
        "train_acc": id_acc + 0.05,
        "id_loss": 1.0 - id_acc,
        "id_acc": id_acc,
        "ood_noise_loss": 1.0 - noise_acc,
        "ood_noise_acc": noise_acc,
        "ood_blur_loss": 1.0 - blur_acc,
        "ood_blur_acc": blur_acc,
        "ood_brightness_loss": 1.0 - brightness_acc,
        "ood_brightness_acc": brightness_acc,
        "avg_ood_acc": avg_ood_acc,
        "id_ood_acc_drop": id_acc - avg_ood_acc,
    }


if __name__ == "__main__":
    unittest.main()
