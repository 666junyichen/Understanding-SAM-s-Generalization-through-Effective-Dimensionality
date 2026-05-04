"""Utility helpers shared by the Part A experiment scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import CHECKPOINT_DIR, DATA_DIR, FIGURES_DIR, LOGS_DIR, RESULTS_DIR


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dirs() -> None:
    """Create standard output directories used by the experiments."""
    for path in (DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Save a model or experiment checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary."""
    return torch.load(Path(path), map_location=map_location)
