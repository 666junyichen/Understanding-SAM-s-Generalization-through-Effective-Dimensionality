"""Sharpness-Aware Minimization optimizer wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization using a PyTorch base optimizer.

    SAM performs two backward passes per batch. The first pass finds a nearby
    high-loss point, and the second pass updates parameters from that point.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid rho, expected non-negative value: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Move parameters to the local worst-case point."""
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for param in group["params"]:
                if param.grad is None:
                    continue

                self.state[param]["old_p"] = param.data.clone()
                perturbation = self._parameter_scale(param, group) * param.grad * scale.to(param)
                param.add_(perturbation)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore original parameters and apply the base optimizer update."""
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.data = self.state[param]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> None:
        """Use first_step and second_step directly in the training loop."""
        if closure is None:
            raise RuntimeError("SAM requires a closure or explicit first_step/second_step calls.")

        closure = torch.enable_grad()(closure)
        closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on the wrapped base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = self._parameter_scale(param, group) * param.grad
                norms.append(torch.norm(grad, p=2).to(shared_device))

        if not norms:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)

    @staticmethod
    def _parameter_scale(param: torch.nn.Parameter, group: dict[str, Any]) -> torch.Tensor:
        if group["adaptive"]:
            return torch.pow(param, 2)
        return torch.ones_like(param)
