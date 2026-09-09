"""Data setup utilities for preparing PyTorch DataLoaders."""

from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np

import torch
from torchvision import transforms

from why_do_I_keep_losing.models.hero_pic_dataset import DotaHeroPicViTDataset


def create_dataloaders(
    parquet_dir: str,
    icons_dir: str,
    transform: Optional[transforms.Compose] = None,
    random_order: bool = False,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[
    DataLoader,
    DataLoader,
    Optional[DataLoader],
    Dataset,
    Dataset,
    Optional[Dataset],
]:
    """Creates training, validation, and optional test DataLoaders,
    using STRATIFIED splitting so each split preserves the overall
    radiant/dire win-rate balance.
    """
    full_dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
        transform=transform,
        random_order=random_order,
    )

    labels = (full_dataset.df["winning_team"] == "radiant").astype(int).values
    all_indices = np.arange(len(full_dataset))

    if test_split > 0:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_split,
            stratify=labels,
            random_state=seed,
        )
        # Recompute val fraction relative to the remaining train+val pool
        val_frac_of_remaining = val_split / (1 - test_split)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_frac_of_remaining,
            stratify=labels[train_val_idx],
            random_state=seed,
        )
    else:
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=val_split,
            stratify=labels,
            random_state=seed,
        )
        test_idx = None

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx) if test_idx is not None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
