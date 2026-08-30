import os
import random 

import pandas as pd
from PIL import Image
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms 
import torchvision.transforms.functional as F


class DotaHeroPicViTDataset(Dataset):
    def __init__(self, parquet_dir: str, icons_dir: str, transform=None, random_order: bool=False):
        self.random_order = random_order
        parquet_dir = Path(parquet_dir)
        parquet_files = sorted(parquet_dir.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {parquet_dir}"
            )

        print(
            f"[INFO] Found {len(parquet_files)} parquet files"
        )

        dfs = [
            pd.read_parquet(parquet_file)
            for parquet_file in parquet_files 
        ]


        self.df = pd.concat(dfs, ignore_index=True)

        duplicate_count = self.df["match_id"].duplicated().sum()

        if duplicate_count > 0:
            print(
                f"[WARNING] Found {duplicate_count} duplicate matches"
            )

            self.df = self.df.drop_duplicates(
                subset="match_id",
                keep="first",
            ).reset_index(drop=True)

        print(
            f"[INFO] Loaded {len(self.df)} unique matches"
        )

        self.icons_dir = icons_dir
        self.transform = transform

        metadata_path = os.path.join(
            os.path.dirname(self.icons_dir),
            "metadata.json",
        )

        with open(metadata_path, "r") as f:
            self.hero_metadata = json.load(f)

        self.hero_tensor_cache={}
        self._pretransform_icons()


    def _pretransform_icons(self):
        """Load and transform all hero icons into memory."""
        for hero_id_str, hero_data in self.hero_metadata.items():
            hero_id = int(hero_id_str)
            localized_name = hero_data["localized_name"]
            # Convert:
            # Anti-Mage       -> anti_mage.png
            # Ancient Apparition -> ancient_apparition.png
            # Nature's Prophet -> natures_prophet.png
            filename = (
                localized_name
                .lower()
                .replace(" ", "_")
                .replace("-", "")
                .replace("'", "")
                + ".png"
            )

            img_path = os.path.join(
                self.icons_dir,
                filename,
            )

            if not os.path.exists(img_path):
                print(
                    f"[WARNING] Icon not found for "
                    f"{localized_name}: {filename}"
                )
                continue

            image = Image.open(img_path).convert("RGB")

            if self.transform:
                tensor = self.transform(image)
            else:
                tensor = F.to_tensor(image)

            self.hero_tensor_cache[hero_id] = tensor

        print(
            f"[INFO] Loaded "
            f"{len(self.hero_tensor_cache)} / "
            f"{len(self.hero_metadata)} hero icons"
        )
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get first hero tensor to infer shape & device for fallbacks
        default_tensor = next(iter(self.hero_tensor_cache.values())).clone().zero_()
        
        radiant_heroes = list(row["radiant_heroes"])
        dire_heroes = list(row["dire_heroes"])

        if self.random_order:
            random.shuffle(radiant_heroes)
            random.shuffle(dire_heroes)
        else:
            radiant_heroes = sorted(
                row["radiant_heroes"],
                key=lambda x: x["role"] if x["role"] is not None else 99,
            )


            dire_heroes = sorted(
                row["dire_heroes"],
                key=lambda x: x["role"] if x["role"] is not None else 99,
            )

        # Retrieve 5 Radiant tensors and 5 Dire tensors
        radiant_tensors = [
            self.hero_tensor_cache.get(p["hero_id"], default_tensor)
            for p in radiant_heroes[:5]
        ]
        dire_tensors = [
            self.hero_tensor_cache.get(p["hero_id"], default_tensor)
            for p in dire_heroes[:5]
        ]

        # 1. Concatenate horizontally to make 2 rows of 5 hero cards each -> Shape: [3, H, 5*W]
        radiant_row = torch.cat(radiant_tensors, dim=2)
        dire_row = torch.cat(dire_tensors, dim=2)

        # 2. Concatenate vertically -> Shape: [3, 2*H, 5*W]
        grid_tensor = torch.cat([radiant_row, dire_row], dim=1)

        # 3. Resize final composite grid to the exact dimensions expected by standard ViT
        match_tensor = F.resize(grid_tensor, [224, 224], antialias=True)

        # Binary label: 1.0 for Radiant win, 0.0 for Dire win
        label = torch.tensor(
            1.0 if row["winning_team"] == "radiant" else 0.0,
            dtype=torch.float32,
        )

        return match_tensor, label

