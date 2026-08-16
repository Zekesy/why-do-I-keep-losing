"""Data setup utilities for preparing PyTorch DataLoaders."""

from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from why_do_I_keep_losing.models.dataset import DotaHeroPicViTDataset


def create_dataloaders(
    parquet_path: str,
    icons_dir: str,
    transform: Optional[transforms.Compose] = None,
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
    """Creates training, validation, and optional test DataLoaders.

    Args:
        parquet_path: Path to the processed matches Parquet file.
        icons_dir: Path to the hero icon PNG files directory.
        transform: Torchvision or timm transform pipeline for hero images.
        batch_size: Number of samples per batch.
        val_split: Fraction of dataset for validation (e.g., 0.15 for 15%).
        test_split: Fraction of dataset for testing (e.g., 0.15 for 15%). Set to
          0.0 to skip.
        num_workers: Number of subprocesses for data loading.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple containing (train_loader, val_loader, test_loader, train_dataset,
        val_dataset, test_dataset)
    """
    full_dataset = DotaHeroPicViTDataset(
        parquet_path=parquet_path,
        icons_dir=icons_dir,
        transform=transform,
    )

    total_len = len(full_dataset)
    test_size = int(total_len * test_split) if test_split > 0 else 0
    val_size = int(total_len * val_split)
    train_size = total_len - val_size - test_size

    # Split dataset reproducibly
    splits = [train_size, val_size]
    if test_size > 0:
        splits.append(test_size)

    subsets = random_split(
        full_dataset,
        splits,
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataset = subsets[0]
    val_dataset = subsets[1]
    test_dataset = subsets[2] if test_size > 0 else None

    # Construct DataLoaders
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
            shuffle=False,  # Keep evaluation deterministic
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
