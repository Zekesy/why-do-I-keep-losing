from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn

from why_do_I_keep_losing.engine.train import eval_step, train_step


class DummyModel(nn.Module):
    """Simple linear model outputting binary logits for testing."""

    def __init__(self):
        super().__init__()
        # Input shape matches [batch_size, 3, 224, 224], outputs scalar logit per sample
        self.fc = nn.Linear(3 * 224 * 224, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def dummy_setup():
    """Provides a dummy model, dataloader, loss function, and optimizer."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = DummyModel().to(device)

    # Synthetic batch: 4 samples of shape [3, 224, 224] with binary targets [0.0 or 1.0]
    X_dummy = torch.randn(4, 3, 224, 224)
    y_dummy = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_dummy, y_dummy)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    return model, dataloader, loss_fn, optimizer, device


def test_train_step(dummy_setup):
    """Verifies train_step performs backprop and returns valid loss and accuracy."""
    model, dataloader, loss_fn, optimizer, device = dummy_setup

    # Record initial weights to verify parameters update
    initial_param = next(model.parameters()).clone()

    train_loss, train_acc = train_step(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    updated_param = next(model.parameters())

    # Assert loss and accuracy types and bounds
    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)
    assert train_loss > 0.0
    assert 0.0 <= train_acc <= 1.0

    # Verify that backpropagation occurred and model weights updated
    assert not torch.equal(initial_param, updated_param)


def test_eval_step(dummy_setup):
    """Verifies eval_step computes metrics without modifying model weights."""
    model, dataloader, loss_fn, _, device = dummy_setup

    initial_param = next(model.parameters()).clone()

    test_loss, test_acc = eval_step(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    updated_param = next(model.parameters())

    # Assert metrics types and valid bounds
    assert isinstance(test_loss, float)
    assert isinstance(test_acc, float)
    assert test_loss > 0.0
    assert 0.0 <= test_acc <= 1.0

    # Verify that eval_step ran in evaluation mode without updating weights
    assert torch.equal(initial_param, updated_param)


def test_train_step_handles_zero_accuracy(dummy_setup):
    """Verifies accuracy calculation when all predictions are completely wrong."""
    model, dataloader, loss_fn, optimizer, device = dummy_setup

    # Mock model to always output large positive logits (+10.0 -> prob ~1.0)
    mock_model = MagicMock()
    mock_model.side_effect = lambda x: torch.full((x.size(0), 1), 10.0, requires_grad=True)
    mock_model.train = MagicMock()

    # Targets are all 0.0 (so predictions will be 0% accurate)
    X_dummy = torch.randn(4, 3, 224, 224)
    y_dummy = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_dummy, y_dummy), batch_size=2
    )

    _, train_acc = train_step(
        model=mock_model,
        dataloader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    assert train_acc == 0.0
