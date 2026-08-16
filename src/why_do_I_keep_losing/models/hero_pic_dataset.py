import os

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms 
import torchvision.transforms.functional as F


class DotaHeroPicViTDataset(Dataset):
    def __init__(self, parquet_path: str, icons_dir: str, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.icons_dir = icons_dir
        self.transform = transform

        self.hero_tensor_cache={}
        self._pretransform_icons()


    def _pretransform_icons(self):
        for filename in os.listdir(self.icons_dir):
            if filename.endswith(".png"):
                hero_id = int(filename.split(".")[0])
                img_path = os.path.join(self.icons_dir, filename)
                image = Image.open(img_path).convert("RGB")

                if self.transform:
                    tensor = self.transform(image)
                else:
                   tensor = F.to_tensor(image) 
                self.hero_tensor_cache[hero_id] = tensor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get first hero tensor to infer shape & device for fallbacks
        default_tensor = next(iter(self.hero_tensor_cache.values())).clone().zero_()

        # Retrieve 5 Radiant tensors and 5 Dire tensors
        radiant_tensors = [
            self.hero_tensor_cache.get(p["hero_id"], default_tensor)
            for p in row["radiant_heroes"][:5]
        ]
        dire_tensors = [
            self.hero_tensor_cache.get(p["hero_id"], default_tensor)
            for p in row["dire_heroes"][:5]
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

