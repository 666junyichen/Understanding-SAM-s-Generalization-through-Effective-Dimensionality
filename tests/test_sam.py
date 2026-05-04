import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class SAMTests(unittest.TestCase):
    def test_first_step_perturbs_and_second_step_updates_parameters(self):
        from sam import SAM

        torch.manual_seed(0)
        model = nn.Linear(2, 1)
        criterion = nn.MSELoss()
        optimizer = SAM(
            model.parameters(),
            base_optimizer=torch.optim.SGD,
            lr=0.1,
            rho=0.05,
        )
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0], [2.0]])

        initial_weight = model.weight.detach().clone()
        criterion(model(inputs), targets).backward()
        optimizer.first_step(zero_grad=True)
        perturbed_weight = model.weight.detach().clone()

        self.assertFalse(torch.allclose(initial_weight, perturbed_weight))
        self.assertIsNone(model.weight.grad)

        criterion(model(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)
        updated_weight = model.weight.detach().clone()

        self.assertFalse(torch.allclose(initial_weight, updated_weight))
        self.assertFalse(torch.allclose(perturbed_weight, updated_weight))
        self.assertIsNone(model.weight.grad)


if __name__ == "__main__":
    unittest.main()
