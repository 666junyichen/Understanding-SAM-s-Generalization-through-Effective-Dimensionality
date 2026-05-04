import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class EvaluateTests(unittest.TestCase):
    def test_evaluate_returns_loss_and_accuracy(self):
        from evaluate import evaluate

        model = nn.Linear(2, 2)
        loader = _toy_loader()
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate(model, loader, criterion, torch.device("cpu"))

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_evaluate_all_returns_id_ood_and_drop_metrics(self):
        from evaluate import evaluate_all

        model = nn.Linear(2, 2)
        loader = _toy_loader()
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate_all(
            model,
            id_loader=loader,
            ood_loaders={"noise": loader, "blur": loader, "brightness": loader},
            criterion=criterion,
            device=torch.device("cpu"),
        )

        expected_keys = {
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
        }
        self.assertTrue(expected_keys.issubset(metrics.keys()))
        self.assertAlmostEqual(
            metrics["id_ood_acc_drop"],
            metrics["id_acc"] - metrics["avg_ood_acc"],
        )


def _toy_loader() -> DataLoader:
    inputs = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)


if __name__ == "__main__":
    unittest.main()
