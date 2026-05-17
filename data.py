"""CIFAR-10 dataloaders and ID/OOD transforms for Part A."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from PIL import ImageEnhance
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    BLUR_KERNEL_SIZE,
    BRIGHTNESS_FACTOR,
    CIFAR10_MEAN,
    CIFAR10_STD,
    DATA_DIR,
    NOISE_STD,
    NUM_WORKERS,
    OOD_CORRUPTIONS,
    SEED,
    VAL_FRACTION,
)


class AddGaussianNoise:
    """Add Gaussian noise to a tensor image before normalization."""

    def __init__(self, std: float = NOISE_STD, mean: float = 0.0) -> None:
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_cifar10_normalization() -> transforms.Normalize:
    """Return the standard CIFAR-10 normalization transform."""
    return transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def get_train_transform() -> transforms.Compose:
    """Return the CIFAR-10 training transform with light augmentation."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            get_cifar10_normalization(),
        ]
    )


def get_id_test_transform() -> transforms.Compose:
    """Return the deterministic in-distribution test transform."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            get_cifar10_normalization(),
        ]
    )


def _adjust_brightness(image: Any) -> Any:
    return ImageEnhance.Brightness(image).enhance(BRIGHTNESS_FACTOR)


def get_ood_transform(corruption: str) -> transforms.Compose:
    """Return a deterministic OOD transform for the CIFAR-10 test set."""
    corruption = _normalize_corruption_name(corruption)

    if corruption == "noise":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                AddGaussianNoise(NOISE_STD),
                get_cifar10_normalization(),
            ]
        )
    if corruption == "blur":
        return transforms.Compose(
            [
                transforms.GaussianBlur(kernel_size=BLUR_KERNEL_SIZE),
                transforms.ToTensor(),
                get_cifar10_normalization(),
            ]
        )
    if corruption == "brightness":
        return transforms.Compose(
            [
                transforms.Lambda(_adjust_brightness),
                transforms.ToTensor(),
                get_cifar10_normalization(),
            ]
        )

    raise ValueError(f"Unknown OOD corruption: {corruption}")


def get_train_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    download: bool = True,
    subset_size: int | None = None,
    dataset_factory: Callable[..., Dataset] = datasets.CIFAR10,
) -> DataLoader:
    """Return the CIFAR-10 training dataloader."""
    dataset = dataset_factory(
        root=str(DATA_DIR),
        train=True,
        download=download,
        transform=get_train_transform(),
    )
    dataset = _maybe_subset(dataset, subset_size)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def get_train_val_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    download: bool = True,
    subset_size: int | None = None,
    val_fraction: float = VAL_FRACTION,
    dataset_factory: Callable[..., Dataset] = datasets.CIFAR10,
) -> tuple[DataLoader, DataLoader]:
    """Return train and validation loaders split from CIFAR-10 training data."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    train_dataset = dataset_factory(
        root=str(DATA_DIR),
        train=True,
        download=download,
        transform=get_train_transform(),
    )
    val_dataset = dataset_factory(
        root=str(DATA_DIR),
        train=True,
        download=download,
        transform=get_id_test_transform(),
    )

    if subset_size is not None:
        train_dataset = _maybe_subset(train_dataset, subset_size)
        val_dataset = _maybe_subset(val_dataset, subset_size)

    val_size = max(1, int(len(train_dataset) * val_fraction))
    train_size = len(train_dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no training examples.")

    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = random_split(
        range(len(train_dataset)),
        [train_size, val_size],
        generator=generator,
    )
    train_subset = Subset(train_dataset, train_indices.indices)
    val_subset = Subset(val_dataset, val_indices.indices)

    return (
        DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        ),
        DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    )


def get_id_test_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    download: bool = True,
    subset_size: int | None = None,
    dataset_factory: Callable[..., Dataset] = datasets.CIFAR10,
) -> DataLoader:
    """Return the CIFAR-10 in-distribution test dataloader."""
    dataset = dataset_factory(
        root=str(DATA_DIR),
        train=False,
        download=download,
        transform=get_id_test_transform(),
    )
    dataset = _maybe_subset(dataset, subset_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_ood_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    download: bool = True,
    subset_size: int | None = None,
    dataset_factory: Callable[..., Dataset] = datasets.CIFAR10,
) -> dict[str, DataLoader]:
    """Return CIFAR-10 test dataloaders with deterministic OOD corruptions."""
    loaders = {}
    for corruption in OOD_CORRUPTIONS:
        short_name = _normalize_corruption_name(corruption)
        dataset = dataset_factory(
            root=str(DATA_DIR),
            train=False,
            download=download,
            transform=get_ood_transform(short_name),
        )
        dataset = _maybe_subset(dataset, subset_size)
        loaders[short_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def _normalize_corruption_name(corruption: str) -> str:
    aliases = {
        "gaussian_noise": "noise",
        "noise": "noise",
        "blur": "blur",
        "brightness": "brightness",
    }
    try:
        return aliases[corruption.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown OOD corruption '{corruption}'. Valid options: {valid}") from exc


def _maybe_subset(dataset: Dataset, subset_size: int | None) -> Dataset:
    if subset_size is None:
        return dataset
    if subset_size <= 0:
        raise ValueError("subset_size must be positive when provided.")
    return Subset(dataset, range(min(subset_size, len(dataset))))
