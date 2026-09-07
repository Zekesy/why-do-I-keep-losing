import torch
import torchvision.transforms.functional as F

def create_match_tensor(
    radiant_heroes,
    dire_heroes,
    hero_tensor_cache,
    canvas_size: int =224,
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
    grid = torch.cat([radiant_row, dire_row], dim=1)

    _, grid_h, grid_w = grid.shape

    if grid_h > canvas_size or grid_w > canvas_size:
        raise ValueError(
            f"Grid ({grid_h}x{grid_w}) exceeds canvas_size={canvas_size}. "
            "Shrink the per-icon cell size in the dataset transform."
        )

    # Center the grid on a zero (black) canvas instead of resizing/blurring it
    canvas = torch.zeros(3, canvas_size, canvas_size)
    top = (canvas_size - grid_h) // 2
    left = (canvas_size - grid_w) // 2
    canvas[:, top: top + grid_h, left: left + grid_w] = grid

    return canvas 

