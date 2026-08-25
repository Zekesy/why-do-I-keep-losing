import torch
import matplotlib.pyplot as plt


def visualize_hero_attributions(
    image,
    attribution,
    hero_scores,
    logit,
):
    """
    Visualize attribution over the 2x5 hero grid.
    """

    probability = torch.sigmoid(
        torch.tensor(logit)
    ).item()

    # Undo ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    display_image = image.cpu() * std + mean
    display_image = display_image.clamp(0, 1)
    display_image = display_image.permute(1, 2, 0)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Original hero grid
    ax.imshow(display_image)

    # Attribution heatmap
    heatmap = ax.imshow(
        attribution,
        cmap="jet",
        alpha=0.45,
    )

    hero_width = 224 / 5

    # Vertical boundaries
    for i in range(1, 5):
        ax.axvline(
            i * hero_width,
            linewidth=1,
            linestyle="--",
        )

    # Radiant/Dire boundary
    ax.axhline(
        112,
        linewidth=2,
    )

    # Label positions
    for i in range(5):

        x = i * hero_width + 2

        radiant_score = hero_scores["radiant"][i]["mean"]
        dire_score = hero_scores["dire"][i]["mean"]

        ax.text(
            x,
            15,
            f"R{i + 1}\n{radiant_score:.3f}",
            fontsize=9,
            color="white",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
            ),
        )

        ax.text(
            x,
            127,
            f"D{i + 1}\n{dire_score:.3f}",
            fontsize=9,
            color="white",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
            ),
        )

    ax.set_title(
        f"Radiant win probability: {probability:.2%}"
    )

    ax.axis("off")

    fig.colorbar(
        heatmap,
        ax=ax,
        label="Attribution",
    )

    plt.tight_layout()
    plt.show()

def get_hero_attributions(
    model,
    image,
    device,
):
    """
    Calculate gradient × input attribution for the
    2x5 Radiant/Dire hero grid.

    Grid layout:

        R1 | R2 | R3 | R4 | R5
        D1 | D2 | D3 | D4 | D5

    Args:
        model: Trained HeroPicViT.
        image: Tensor [3, 224, 224].
        device: torch.device.

    Returns:
        attribution: [224, 224] attribution map.
        hero_scores: Attribution scores for R1-R5 and D1-D5.
        logit: Raw model output.
    """

    model.eval()

    x = image.unsqueeze(0).to(device)
    x.requires_grad_(True)

    # Forward pass
    logit = model(x)

    # Calculate gradient of the Radiant-win logit
    model.zero_grad()
    logit.backward()

    # Gradient × input
    attribution = (x.grad * x).abs().sum(dim=1)

    # [1, 224, 224] -> [224, 224]
    attribution = attribution.squeeze(0).detach().cpu()

    # Normalize
    attribution = (
        attribution - attribution.min()
    ) / (
        attribution.max() - attribution.min() + 1e-8
    )

    hero_scores = {
        "radiant": [],
        "dire": [],
    }

    hero_width = 224 / 5
    hero_height = 224 / 2

    for team in ["radiant", "dire"]:

        y1 = 0 if team == "radiant" else 112
        y2 = 112 if team == "radiant" else 224

        for i in range(5):

            x1 = int(round(i * hero_width))
            x2 = int(round((i + 1) * hero_width))

            region = attribution[y1:y2, x1:x2]

            hero_scores[team].append({
                "position": i + 1,
                "mean": region.mean().item(),
                "sum": region.sum().item(),
                "max": region.max().item(),
            })

    return attribution, hero_scores, logit.item()
