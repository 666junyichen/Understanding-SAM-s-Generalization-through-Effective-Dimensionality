"""Model definitions for Part A experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18

from config import MODEL_NAME, NUM_CLASSES


def get_model(model_name: str = MODEL_NAME, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Return a model for CIFAR-10 classification."""
    model_name = model_name.lower()

    if model_name == "resnet18":
        return get_cifar10_resnet18(num_classes=num_classes)

    raise ValueError(f"Unknown model name: {model_name}")


def get_cifar10_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Create a ResNet18 adapted for 32x32 CIFAR-10 images."""
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
