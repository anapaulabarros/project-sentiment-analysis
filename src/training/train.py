"""Training pipeline: data splitting, tensor conversion, DataLoader creation, and training loop."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import SentimentMLP
from src.utils.config import RANDOM_SEED


def split_dataset(
    data: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float,
) -> tuple:
    """Split the dataset into training and test sets.

    Args:
        data: Preprocessed DataFrame.
        text_column: Name of the text column.
        label_column: Name of the label column.
        test_size: Proportion of the dataset to use for testing (0–1).

    Returns:
        Tuple of (x_train, x_test, y_train, y_test).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        data[text_column],
        data[label_column],
        test_size=test_size,
        random_state=RANDOM_SEED,
    )
    return x_train, x_test, y_train, y_test


def to_tensors(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert NumPy arrays to PyTorch tensors.

    Args:
        X: Feature matrix.
        y: Label array.

    Returns:
        Tuple of (feature tensor, label tensor).
    """
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def create_dataloader(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap tensors into a DataLoader.

    Args:
        X_tensor: Feature tensor.
        y_tensor: Label tensor.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.

    Returns:
        Configured DataLoader.
    """
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: SentimentMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Args:
        model: SentimentMLP to train.
        loader: Training DataLoader.
        optimizer: Gradient optimizer.
        criterion: Loss function.
        device: Target device.

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(
    model: SentimentMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch.

    Args:
        model: SentimentMLP to evaluate.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Target device.

    Returns:
        Mean validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch).squeeze(1)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)


def run_training(
    model: SentimentMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
) -> SentimentMLP:
    """Execute the full training loop with validation.

    Args:
        model: SentimentMLP to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Gradient optimizer.
        criterion: Loss function.
        num_epochs: Number of training epochs.
        device: Target device.

    Returns:
        Trained SentimentMLP.
    """
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:>2}/{num_epochs} — train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
    return model
