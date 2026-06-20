"""Unit tests for the training module."""

import unittest

import numpy as np
import torch

from src.training.train import create_dataloader, to_tensors


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


if __name__ == "__main__":
    unittest.main()
