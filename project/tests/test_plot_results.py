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


if __name__ == "__main__":
    unittest.main()
