import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class ModelTests(unittest.TestCase):
    def test_cifar10_resnet18_outputs_ten_class_logits(self):
        from models import get_model

        model = get_model("resnet18", num_classes=10)
        model.eval()

        with torch.no_grad():
            logits = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(tuple(logits.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
