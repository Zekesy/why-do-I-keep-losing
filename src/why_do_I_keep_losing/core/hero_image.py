import torch
import torchvision.transforms.functional as F

def create_match_tensor(
    radiant_heroes,
    dire_heroes,
    hero_tensor_cache,
):
    default_tensor = next(iter(hero_tensor_cache.values())).clone().zero_()

    radiant_tensors = [
        hero_tensor_cache.get(hero_id, default_tensor)
        for hero_id in radiant_heroes[:5]
    ]
    dire_tensors = [
        hero_tensor_cache.get(hero_id, default_tensor)
        for hero_id in dire_heroes[:5]
    ]

    # 1. Concatenate horizontally to make 2 rows of 5 hero cards each -> Shape: [3, H, 5*W]
    radiant_row = torch.cat(radiant_tensors, dim=2)
    dire_row = torch.cat(dire_tensors, dim=2)

    # 2. Concatenate vertically -> Shape: [3, 2*H, 5*W]
    grid_tensor = torch.cat([radiant_row, dire_row], dim=1)

    # 3. Resize final composite grid to the exact dimensions expected by standard ViT
    match_tensor = F.resize(grid_tensor, [224, 224], antialias=True)
    
    return match_tensor

