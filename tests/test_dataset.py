import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from why_do_I_keep_losing.models.hero_pic_dataset import (
    DotaHeroPicViTDataset,
)


@pytest.fixture
def mock_dataset_files(tmp_path):
    """Create temporary dummy hero icons, metadata, and Parquet files."""

    # Directory structure:
    #
    # tmp_path/
    # ├── metadata.json
    # ├── hero_icons/
    # │   ├── hero_1.png
    # │   ├── ...
    # │   └── hero_10.png
    # └── parquet/
    #     └── test_pro_matches.parquet

    icons_dir = tmp_path / "hero_icons"
    parquet_dir = tmp_path / "parquet"

    icons_dir.mkdir()
    parquet_dir.mkdir()

    # ---------------------------------------------------------
    # Create mock hero metadata and icons
    # ---------------------------------------------------------

    hero_metadata = {}

    for hero_id in range(1, 11):
        hero_name = f"Hero {hero_id}"

        hero_metadata[str(hero_id)] = {
            "localized_name": hero_name,
        }

        # This must match the filename logic in the Dataset:
        #
        # "Hero 1" -> "hero_1.png"
        filename = f"hero_{hero_id}.png"

        dummy_img = Image.fromarray(
            np.uint8(
                np.random.rand(100, 100, 3) * 255
            )
        )

        dummy_img.save(
            icons_dir / filename
        )

    # Dataset expects metadata.json one directory above icons_dir
    metadata_path = tmp_path / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(hero_metadata, f)

    # ---------------------------------------------------------
    # Create mock match data
    # ---------------------------------------------------------

    mock_data = {
        "match_id": [12345, 67890],

        "winning_team": [
            "radiant",
            "dire",
        ],

        "radiant_heroes": [
            [
                {"hero_id": 1, "role": 1},
                {"hero_id": 2, "role": 2},
                {"hero_id": 3, "role": 3},
                {"hero_id": 4, "role": 4},
                {"hero_id": 5, "role": 5},
            ],
            [
                {"hero_id": 1, "role": 1},
                {"hero_id": 2, "role": 2},
                {"hero_id": 3, "role": 3},
                {"hero_id": 4, "role": 4},
                {"hero_id": 5, "role": 5},
            ],
        ],

        "dire_heroes": [
            [
                {"hero_id": 6, "role": 1},
                {"hero_id": 7, "role": 2},
                {"hero_id": 8, "role": 3},
                {"hero_id": 9, "role": 4},
                {"hero_id": 10, "role": 5},
            ],
            [
                {"hero_id": 6, "role": 1},
                {"hero_id": 7, "role": 2},
                {"hero_id": 8, "role": 3},
                {"hero_id": 9, "role": 4},
                {"hero_id": 10, "role": 5},
            ],
        ],
    }

    df = pd.DataFrame(mock_data)

    parquet_path = parquet_dir / "test_pro_matches.parquet"

    df.to_parquet(parquet_path)

    return str(parquet_dir), str(icons_dir)


# =============================================================
# Dataset tests
# =============================================================


def test_dataset_type(mock_dataset_files):
    """Verify that the created object is a PyTorch Dataset."""

    parquet_dir, icons_dir = mock_dataset_files

    dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
    )

    assert isinstance(dataset, Dataset)
    assert isinstance(dataset, DotaHeroPicViTDataset)


def test_dataset_length(mock_dataset_files):
    """Verify that all matches in the Parquet directory are loaded."""

    parquet_dir, icons_dir = mock_dataset_files

    dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
    )

    assert len(dataset) == 2


def test_dota_dataset_output_shapes_and_types(
    mock_dataset_files,
):
    """Verify __getitem__ returns the expected tensor shape and types."""

    parquet_dir, icons_dir = mock_dataset_files

    custom_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
        transform=custom_transform,
    )

    match_tensor, label = dataset[0]

    # ---------------------------------------------------------
    # Tensor checks
    # ---------------------------------------------------------

    assert isinstance(match_tensor, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert match_tensor.shape == torch.Size(
        [3, 224, 224]
    )

    assert match_tensor.dtype == torch.float32

    # ---------------------------------------------------------
    # Label checks
    # ---------------------------------------------------------

    assert label.shape == torch.Size([])
    assert label.dtype == torch.float32

    # First match is a Radiant win
    assert label.item() == 1.0


def test_dota_dataset_label_encoding(
    mock_dataset_files,
):
    """Verify that a Dire win is encoded as 0.0."""

    parquet_dir, icons_dir = mock_dataset_files

    dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
    )

    _, label_dire_win = dataset[1]

    assert label_dire_win.item() == 0.0


# =============================================================
# random_order tests
# =============================================================


def test_random_order_default_is_false(
    mock_dataset_files,
):
    """Verify random_order defaults to False."""

    parquet_dir, icons_dir = mock_dataset_files

    dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
    )

    assert dataset.random_order is False


def test_random_order_can_be_enabled(
    mock_dataset_files,
):
    """Verify random_order=True is accepted and stored."""

    parquet_dir, icons_dir = mock_dataset_files

    dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
        random_order=True,
    )

    assert dataset.random_order is True


def test_random_order_output(
    mock_dataset_files,
):
    """Verify random ordering still produces a valid model input."""

    parquet_dir, icons_dir = mock_dataset_files

    dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
        random_order=True,
    )

    match_tensor, label = dataset[0]

    assert isinstance(match_tensor, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert match_tensor.shape == torch.Size(
        [3, 224, 224]
    )

    assert match_tensor.dtype == torch.float32
    assert label.dtype == torch.float32


def test_random_order_does_not_change_dataset_length(
    mock_dataset_files,
):
    """Verify random ordering does not change dataset size."""

    parquet_dir, icons_dir = mock_dataset_files

    ordered_dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
        random_order=False,
    )

    random_dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
        random_order=True,
    )

    assert len(ordered_dataset) == len(random_dataset)


def test_random_order_preserves_label(
    mock_dataset_files,
):
    """Verify random ordering does not affect the match label."""

    parquet_dir, icons_dir = mock_dataset_files

    dataset = DotaHeroPicViTDataset(
        parquet_dir=parquet_dir,
        icons_dir=icons_dir,
        random_order=True,
    )

    # Match 0 = Radiant win
    _, label_0 = dataset[0]

    # Match 1 = Dire win
    _, label_1 = dataset[1]

    assert label_0.item() == 1.0
    assert label_1.item() == 0.0
