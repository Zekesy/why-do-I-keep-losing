from pathlib import Path 

from torchvision import transforms
import torch
import matplotlib.pyplot as plt

from why_do_I_keep_losing.utils.data_setup import create_dataloaders

DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "processed" / "pro_matches_8831344795_8948533452.parquet"
ICONS_DIR = DATA_DIR / "heroes" / "icons"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print(f"Parquet: {PARQUET_PATH}")
print(f"Exists: {PARQUET_PATH.exists()}")

print(f"Icons: {ICONS_DIR}")
print(f"Exists: {ICONS_DIR.exists()}")
print(f"Number of icons: {len(list(ICONS_DIR.glob('*.png')))}")

train_loader, val_loader, test_loader, \
train_dataset, val_dataset, test_dataset = create_dataloaders(
    parquet_path=str(PARQUET_PATH),
    icons_dir=str(ICONS_DIR),
    transform=transform,
    batch_size=4,
    val_split=0.15,
    test_split=0.15,
    num_workers=0, 
)

print()
print("Dataset sizes:")
print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))

# ============================================================
# Inspect one match
# ============================================================

sample_idx = 0

image, label = train_dataset[sample_idx]

# train_dataset is a torch.utils.data.Subset
original_idx = train_dataset.indices[sample_idx]

row = train_dataset.dataset.df.iloc[original_idx]


print("\n" + "=" * 60)
print("MATCH")
print("=" * 60)

print(f"Match ID: {row['match_id']}")
print(f"Winner:   {row['winning_team']}")

# Print hero order
radiant = sorted(
    row["radiant_heroes"],
    key=lambda x: x["role"] if x["role"] is not None else 99,
)
dire = sorted(
    row["dire_heroes"],
    key=lambda x: x["role"] if x["role"] is not None else 99,
)

print("\nRadiant:")
for hero in radiant:
    print(
        f"  Role {hero['role']}: "
        f"Hero ID {hero['hero_id']}"
    )
print("\nDire:")
for hero in dire:
    print(
        f"  Role {hero['role']}: "
        f"Hero ID {hero['hero_id']}"
    )

# Check image
print("\n" + "=" * 60)
print("IMAGE")
print("=" * 60)

print("Image shape:", image.shape)
print("Label:", label.item())

assert image.shape == torch.Size([3, 224, 224]), (
    f"Unexpected image shape: {image.shape}"
)


plt.figure(figsize=(10, 5))
plt.imshow(image.permute(1, 2, 0))
plt.axhline(
    y=112,
    linewidth=2,
)
plt.text(
    5,
    10,
    "RADIANT",
    fontsize=14,
    weight="bold",
)
plt.text(
    5,
    122,
    "DIRE",
    fontsize=14,
    weight="bold",
)

plt.axis("off")
plt.title(
    f"Match {row['match_id']} | "
    f"Winner: {row['winning_team']}"
)

plt.show()
