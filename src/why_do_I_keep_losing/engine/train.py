"""Engine module containing PyTorch training and evaluation steps."""

from typing import Dict, List, Tuple
import torch
from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0.0, 0.0
    for batch, (X, y) in enumerate(
        tqdm(dataloader, desc="Training", leave=False)
    ):
        X, y = X.to(device), y.to(device)

        # Forward pass & shape squeeze to match label dimensions [batch_size]
        y_pred = model(X).squeeze(-1)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Binary predictions and accuracy calculation
        probs = torch.sigmoid(y_pred)
        preds = (probs >= 0.5).float()
        train_acc += (preds == y).sum().item() / len(y)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def eval_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0.0, 0.0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(
            tqdm(dataloader, desc="Evaluating", leave=False)
        ):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X).squeeze(-1)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            probs = torch.sigmoid(test_pred_logits)
            preds = (probs >= 0.5).float()
            test_acc += (preds == y).sum().item() / len(y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    model.to(device)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = eval_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"Epoch: {epoch + 1:02d} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
