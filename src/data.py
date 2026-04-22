from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def _select_class_subset(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_a: int,
    class_b: int,
    max_per_class: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx_a = (labels == class_a).nonzero(as_tuple=False).squeeze(1)[:max_per_class]
    idx_b = (labels == class_b).nonzero(as_tuple=False).squeeze(1)[:max_per_class]
    idx = torch.cat([idx_a, idx_b], dim=0)
    x = images[idx]
    y = labels[idx]
    y = torch.where(y == class_a, torch.tensor(0), torch.tensor(1))
    return x, y


def apply_square_trigger(images: torch.Tensor, size: int = 4, value: float = 1.0) -> torch.Tensor:
    out = images.clone()
    out[:, :, :size, :size] = value
    return out


def _select_poison_indices(labels: torch.Tensor, target_label: int, poisoning_k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = labels.shape[0]
    candidates = (labels != target_label).nonzero(as_tuple=False).squeeze(1)
    if candidates.numel() == 0:
        candidates = torch.arange(n)
    k = min(poisoning_k, candidates.numel())
    perm = torch.randperm(candidates.numel(), generator=g)[:k]
    return candidates[perm]


def poison_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    poisoning_k: int,
    target_label: int,
    seed: int,
    trigger_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chosen = _select_poison_indices(y_train, target_label, poisoning_k, seed)

    x_poison = x_train.clone()
    y_poison = y_train.clone()
    x_poison[chosen] = apply_square_trigger(x_poison[chosen], size=trigger_size)
    y_poison[chosen] = target_label
    return x_poison, y_poison


def poison_dataset_svd_lowvar(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    poisoning_k: int,
    target_label: int,
    seed: int,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chosen = _select_poison_indices(y_train, target_label, poisoning_k, seed)

    # Flatten data and find the lowest-variance direction via SVD.
    n = x_train.shape[0]
    x_flat = x_train.view(n, -1)
    x_center = x_flat - x_flat.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x_center, full_matrices=False)
    lowvar_dir = vh[-1]

    if lowvar_dir.sum() < 0:
        lowvar_dir = -lowvar_dir

    direction = lowvar_dir / lowvar_dir.abs().max().clamp(min=1e-8)
    perturbation = direction.view_as(x_train[0]) * float(scale) * 0.20

    x_poison = x_train.clone()
    y_poison = y_train.clone()
    x_poison[chosen] = torch.clamp(x_poison[chosen] + perturbation, 0.0, 1.0)
    y_poison[chosen] = target_label
    return x_poison, y_poison


def create_poisoned_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    poisoning_k: int,
    target_label: int,
    seed: int,
    poison_method: str,
    trigger_size: int,
    svd_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if poison_method == "square":
        return poison_dataset(
            x_train,
            y_train,
            poisoning_k=poisoning_k,
            target_label=target_label,
            seed=seed,
            trigger_size=trigger_size,
        )
    if poison_method == "svd_lowvar":
        return poison_dataset_svd_lowvar(
            x_train,
            y_train,
            poisoning_k=poisoning_k,
            target_label=target_label,
            seed=seed,
            scale=svd_scale,
        )
    raise ValueError(f"Unsupported poison_method={poison_method}")


def load_binary_fashion_mnist(
    data_dir: str,
    max_train_per_class: int,
    max_test_per_class: int,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_path = Path(data_dir) / f"fashion_binary_{max_train_per_class}_{max_test_per_class}.pt"
    if use_cache and cache_path.exists():
        packed = torch.load(cache_path)
        return packed["x_train"], packed["y_train"], packed["x_test"], packed["y_test"]

    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    x_train, y_train = _select_class_subset(
        train_ds.data.unsqueeze(1).float() / 255.0,
        train_ds.targets,
        class_a=0,
        class_b=1,
        max_per_class=max_train_per_class,
    )
    x_test, y_test = _select_class_subset(
        test_ds.data.unsqueeze(1).float() / 255.0,
        test_ds.targets,
        class_a=0,
        class_b=1,
        max_per_class=max_test_per_class,
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
            },
            cache_path,
        )

    return x_train, y_train, x_test, y_test


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)
