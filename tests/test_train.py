import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class TrainTests(unittest.TestCase):
    def test_train_one_epoch_sgd_runs_on_toy_dataset(self):
        from train import train_one_epoch_sgd

        model = nn.Linear(2, 2)
        loader = _toy_loader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        metrics = train_one_epoch_sgd(model, loader, optimizer, criterion, torch.device("cpu"))

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_train_one_epoch_sam_runs_on_toy_dataset(self):
        from sam import SAM
        from train import train_one_epoch_sam

        model = nn.Linear(2, 2)
        loader = _toy_loader()
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=0.1, rho=0.05)
        criterion = nn.CrossEntropyLoss()

        metrics = train_one_epoch_sam(model, loader, optimizer, criterion, torch.device("cpu"))

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


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
