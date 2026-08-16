"""Main training script for the Dota Hero Pick ViT model."""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms

from why_do_I_keep_losing.engine.train import train
from why_do_I_keep_losing.models.hero_pic_ViT import HeroPicViT
from why_do_I_keep_losing.utils.data_setup import create_dataloaders


def main():
    # ---------------------------------------------------------
    # 1. Hyperparameters & Settings
    # ---------------------------------------------------------
    PARQUET_PATH = "data/processed/pro_matches.parquet"
    ICONS_DIR = "data/external/hero_icons"

    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    SEED = 42

    # Auto-detect available compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # ---------------------------------------------------------
    # 2. Define Image Transformations
    # ---------------------------------------------------------
    # Standard ImageNet normalization expected by pre-trained ViTs
    vit_transforms = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # ---------------------------------------------------------
    # 3. Create DataLoaders
    # ---------------------------------------------------------
    print("[*] Creating DataLoaders...")
    train_loader, val_loader, _, _, _, _ = create_dataloaders(
        parquet_path=PARQUET_PATH,
        icons_dir=ICONS_DIR,
        transform=vit_transforms,
        batch_size=BATCH_SIZE,
        val_split=0.15,
        test_split=0.0,
        num_workers=2,
        seed=SEED,
    )

    # ---------------------------------------------------------
    # 4. Instantiate Model, Loss Function, and Optimizer
    # ---------------------------------------------------------
    print("[*] Initializing HeroPicViT model...")
    model = HeroPicViT(
        model_name="vit_base_patch16_224", pretrained=True, drop_rate=0.2
    )

    # Loss function for raw logits output
    loss_fn = nn.BCEWithLogitsLoss()

    # AdamW optimizer with weight decay regularisation
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2
    )

    # ---------------------------------------------------------
    # 5. Run Training Loop
    # ---------------------------------------------------------
    print(f"[*] Starting training for {EPOCHS} epochs...")
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device,
    )

    # ---------------------------------------------------------
    # 6. Save Trained Weights
    # ---------------------------------------------------------
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "hero_pic_vit_latest.pth"

    torch.save(model.state_dict(), save_path)
    print(f"[*] Model successfully saved to {save_path}")


if __name__ == "__main__":
    main()
