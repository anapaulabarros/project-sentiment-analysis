"""Unit tests for the training module."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from src.models.model import SentimentMLP, build_model
from src.training.train import (
    create_dataloader,
    run_training,
    to_tensors,
    train_epoch,
    validate_epoch,
)
from src.utils.config import BATCH_SIZE, HIDDEN_DIM, MAX_FEATURES, OUTPUT_DIM


def _make_loader(n: int = 32, features: int = MAX_FEATURES, batch: int = 16) -> object:
    X = torch.rand(n, features)
    y = torch.randint(0, 2, (n,)).float()
    return create_dataloader(X, y, batch_size=batch, shuffle=False)


class TestToTensors(unittest.TestCase):
    def test_returns_float32_tensors(self) -> None:
        X = np.random.rand(10, 50).astype(np.float32)
        y = np.array([0, 1] * 5, dtype=np.float32)
        X_t, y_t = to_tensors(X, y)
        self.assertEqual(X_t.dtype, torch.float32)
        self.assertEqual(y_t.dtype, torch.float32)

    def test_shapes_preserved(self) -> None:
        X = np.random.rand(20, 100)
        y = np.zeros(20)
        X_t, y_t = to_tensors(X, y)
        self.assertEqual(tuple(X_t.shape), (20, 100))
        self.assertEqual(tuple(y_t.shape), (20,))


class TestCreateDataLoader(unittest.TestCase):
    def test_batch_size(self) -> None:
        X = torch.rand(64, 50)
        y = torch.zeros(64)
        loader = create_dataloader(X, y, batch_size=16)
        batch_X, _ = next(iter(loader))
        self.assertEqual(batch_X.shape[0], 16)

    def test_dataset_length(self) -> None:
        X = torch.rand(100, 50)
        y = torch.zeros(100)
        loader = create_dataloader(X, y, batch_size=10)
        self.assertEqual(len(loader.dataset), 100)


class TestTrainEpoch(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.model = build_model(MAX_FEATURES, HIDDEN_DIM, OUTPUT_DIM).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loader = _make_loader()

    def test_returns_float(self) -> None:
        loss = train_epoch(self.model, self.loader, self.optimizer, self.criterion, self.device)
        self.assertIsInstance(loss, float)

    def test_loss_is_positive(self) -> None:
        loss = train_epoch(self.model, self.loader, self.optimizer, self.criterion, self.device)
        self.assertGreater(loss, 0.0)


class TestValidateEpoch(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.model = build_model(MAX_FEATURES, HIDDEN_DIM, OUTPUT_DIM).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.loader = _make_loader()

    def test_returns_float(self) -> None:
        loss = validate_epoch(self.model, self.loader, self.criterion, self.device)
        self.assertIsInstance(loss, float)

    def test_loss_is_positive(self) -> None:
        loss = validate_epoch(self.model, self.loader, self.criterion, self.device)
        self.assertGreater(loss, 0.0)


class TestRunTraining(unittest.TestCase):
    def test_returns_sentiment_mlp(self) -> None:
        device = torch.device("cpu")
        model = build_model(MAX_FEATURES, HIDDEN_DIM, OUTPUT_DIM).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = _make_loader()
        result = run_training(model, loader, loader, optimizer, criterion, num_epochs=2, device=device)
        self.assertIsInstance(result, SentimentMLP)


if __name__ == "__main__":
    unittest.main()
