import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class RunExperimentTests(unittest.TestCase):
    def test_build_optimizer_creates_sgd_and_sam(self):
        from run_experiment import build_optimizer
        from sam import SAM

        model = nn.Linear(2, 2)

        sgd = build_optimizer("sgd", model)
        sam = build_optimizer("sam", model)

        self.assertIsInstance(sgd, torch.optim.SGD)
        self.assertIsInstance(sam, SAM)

    def test_save_csv_writes_rows_with_requested_fieldnames(self):
        from run_experiment import save_csv

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metrics.csv"
            save_csv(path, [{"optimizer": "sgd", "id_acc": 0.8}], ["optimizer", "id_acc"])

            with path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(rows, [{"optimizer": "sgd", "id_acc": "0.8"}])


if __name__ == "__main__":
    unittest.main()
