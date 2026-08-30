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

from why_do_I_keep_losing.core.hero_image import create_match_tensor

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
        # Convert to Ids to match func input format
        radiant_hero_ids = [
            hero["hero_id"] for hero in radiant_heroes[:5]
        ]
        dire_hero_ids = [
            hero["hero_id"] for hero in dire_heroes[:5]
        ]
        match_tensor = create_match_tensor(radiant_hero_ids, dire_hero_ids, self.hero_tensor_cache)

        # Binary label: 1.0 for Radiant win, 0.0 for Dire win
        label = torch.tensor(
            1.0 if row["winning_team"] == "radiant" else 0.0,
            dtype=torch.float32,
        )

        return match_tensor, label

