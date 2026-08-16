from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torchvision import transforms

from why_do_I_keep_losing.models.hero_pic_dataset import DotaHeroPicViTDataset


@pytest.fixture
def mock_dataset_files(tmp_path):
    """Creates temporary dummy hero icons and a mock Parquet dataset file."""
    icons_dir = tmp_path / "hero_icons"
    icons_dir.mkdir()

    # Generate 10 dummy hero PNG icons (100x100 RGB images)
    hero_ids = list(range(1, 11))
    for hero_id in hero_ids:
        dummy_img = Image.fromarray(
            np.uint8(np.random.rand(100, 100, 3) * 255)
        )
        dummy_img.save(icons_dir / f"{hero_id}.png")

    # Create mock Parquet metadata matching the expected dataset schema
    mock_data = {
        "match_id": [12345, 67890],
        "winning_team": ["radiant", "dire"],
        "radiant_heroes": [
            [{"hero_id": 1}, {"hero_id": 2}, {"hero_id": 3}, {"hero_id": 4}, {"hero_id": 5}],
            [{"hero_id": 1}, {"hero_id": 2}, {"hero_id": 3}, {"hero_id": 4}, {"hero_id": 5}],
        ],
        "dire_heroes": [
            [{"hero_id": 6}, {"hero_id": 7}, {"hero_id": 8}, {"hero_id": 9}, {"hero_id": 10}],
            [{"hero_id": 6}, {"hero_id": 7}, {"hero_id": 8}, {"hero_id": 9}, {"hero_id": 10}],
        ],
    }

    parquet_path = tmp_path / "test_pro_matches.parquet"
    df = pd.DataFrame(mock_data)
    df.to_parquet(parquet_path)

    return str(parquet_path), str(icons_dir)


def test_dota_dataset_output_shapes_and_types(mock_dataset_files):
    """Verifies that __getitem__ returns the expected tensor shapes and data types."""
    parquet_path, icons_dir = mock_dataset_files

    # Custom transform to simulate pre-trained model input specifications
    custom_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    dataset = DotaHeroPicViTDataset(
        parquet_path=parquet_path,
        icons_dir=icons_dir,
        transform=custom_transform,
    )

    # Check dataset length
    assert len(dataset) == 2

    # Fetch first sample
    match_tensor, label = dataset[0]

    # Check tensor types
    assert isinstance(match_tensor, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    # Shape expectations:
    # Channel dimension = 3 (RGB)
    # Height x Width = 224 x 224 (Resized grid output)
    assert match_tensor.shape == torch.Size([3, 224, 224])
    assert match_tensor.dtype == torch.float32

    # Label assertions
    assert label.shape == torch.Size([])  # Scalar tensor
    assert label.dtype == torch.float32
    assert label.item() == 1.0  # Radiant win


def test_dota_dataset_label_encoding(mock_dataset_files):
    """Verifies that Dire wins evaluate to 0.0 binary labels."""
    parquet_path, icons_dir = mock_dataset_files

    dataset = DotaHeroPicViTDataset(
        parquet_path=parquet_path,
        icons_dir=icons_dir,
    )

    _, label_dire_win = dataset[1]
    assert label_dire_win.item() == 0.0
