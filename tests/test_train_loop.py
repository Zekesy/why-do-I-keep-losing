from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from why_do_I_keep_losing.engine.train import train


@pytest.fixture
def mock_train_setup():
    """Provides lightweight dummy objects for model, dataloaders, optimizer, and loss function."""
    model = nn.Linear(10, 1)
    train_loader = MagicMock(spec=torch.utils.data.DataLoader)
    test_loader = MagicMock(spec=torch.utils.data.DataLoader)
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    loss_fn = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    return model, train_loader, test_loader, optimizer, loss_fn, device


@patch("why_do_I_keep_losing.engine.train.eval_step")
@patch("why_do_I_keep_losing.engine.train.train_step")
def test_train_loop_execution_and_results_structure(
    mock_train_step, mock_eval_step, mock_train_setup
):
    """Verifies that train() iterates for the specified number of epochs and records metrics."""
    model, train_loader, test_loader, optimizer, loss_fn, device = (
        mock_train_setup
    )

    # Mock return values for train_step and eval_step: (loss, accuracy)
    mock_train_step.return_value = (0.50, 0.75)
    mock_eval_step.return_value = (0.45, 0.80)

    epochs = 3
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
    )

    # Verify train_step and eval_step were called exactly `epochs` times
    assert mock_train_step.call_count == epochs
    assert mock_eval_step.call_count == epochs

    # Verify the results dictionary structure and lengths
    expected_keys = {"train_loss", "train_acc", "test_loss", "test_acc"}
    assert set(results.keys()) == expected_keys

    for key in expected_keys:
        assert len(results[key]) == epochs

    # Verify recorded metric values match mocked returns
    assert results["train_loss"] == [0.50, 0.50, 0.50]
    assert results["train_acc"] == [0.75, 0.75, 0.75]
    assert results["test_loss"] == [0.45, 0.45, 0.45]
    assert results["test_acc"] == [0.80, 0.80, 0.80]


@patch("why_do_I_keep_losing.engine.train.eval_step")
@patch("why_do_I_keep_losing.engine.train.train_step")
def test_train_loop_passes_correct_arguments(
    mock_train_step, mock_eval_step, mock_train_setup
):
    """Verifies that train() correctly propagates arguments down to train_step and eval_step."""
    model, train_loader, test_loader, optimizer, loss_fn, device = (
        mock_train_setup
    )

    mock_train_step.return_value = (0.6, 0.6)
    mock_eval_step.return_value = (0.6, 0.6)

    train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=1,
        device=device,
    )

    # Assert train_step was called with the exact objects provided
    mock_train_step.assert_called_once_with(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    # Assert eval_step was called with the exact objects provided
    mock_eval_step.assert_called_once_with(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )
