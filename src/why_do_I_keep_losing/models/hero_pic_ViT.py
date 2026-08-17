"""Vision Transformer model wrapper for Dota hero pick binary classification."""

import timm
import torch
import torch.nn as nn


class HeroPicViT(nn.Module):
    """ViT model adapted for predicting match outcome (binary classification)

    from a 2D hero grid tensor.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        """Args:

        model_name: timm ViT backbone architecture name. pretrained: Whether to
        load pre-trained ImageNet weights. drop_rate: Dropout rate before the
        final linear classifier layer.
        """
        super().__init__()

        # Load backbone ViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Removes default head, returning feature representation
            drop_rate=drop_rate,
        )

        # Get feature dimension outputted by ViT backbone (e.g. 768 for ViT-Base)
        in_features = self.backbone.num_features

        # Binary classification head returning single scalar logit per sample
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input grid tensor of shape [batch_size, 3, 224, 224]

        Returns:
            Raw logits tensor of shape [batch_size, 1]
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
