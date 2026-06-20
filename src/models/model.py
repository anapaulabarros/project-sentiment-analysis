"""Sentiment classification model — PyTorch MLP."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SentimentMLP(nn.Module):
    """Binary sentiment classifier: one hidden layer MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


def build_model(input_dim: int, hidden_dim: int, output_dim: int) -> SentimentMLP:
    """Instantiate the sentiment MLP.

    Args:
        input_dim: Number of input features (vocabulary size).
        hidden_dim: Number of hidden units.
        output_dim: Number of output units (1 for binary classification).

    Returns:
        Initialized SentimentMLP.
    """
    return SentimentMLP(input_dim, hidden_dim, output_dim)


def save_model(model: SentimentMLP, path: str) -> None:
    """Persist model weights to disk.

    Args:
        model: Trained SentimentMLP.
        path: Destination file path (.pt).
    """
    torch.save(model.state_dict(), path)


def load_model(
    path: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> SentimentMLP:
    """Load model weights from disk.

    Args:
        path: Path to the saved state dict (.pt).
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.

    Returns:
        SentimentMLP in eval mode.
    """
    model = SentimentMLP(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict(
    model: SentimentMLP,
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Generate binary predictions from a tensor.

    Args:
        model: Trained SentimentMLP.
        x: Input tensor of shape (n_samples, input_dim).
        device: Target device.

    Returns:
        NumPy array of binary labels (0 or 1).
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits)
        labels = (probs >= 0.5).long()
    return labels.cpu().numpy()
