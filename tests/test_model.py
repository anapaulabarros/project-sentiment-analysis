"""Unit tests for the model module."""

import os
import tempfile
import unittest

import numpy as np
import torch

from src.models.model import build_model, load_model, predict, save_model
from src.utils.config import HIDDEN_DIM, MAX_FEATURES, OUTPUT_DIM


class TestSentimentMLP(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.model = build_model(MAX_FEATURES, HIDDEN_DIM, OUTPUT_DIM).to(self.device)

    def test_forward_output_shape(self) -> None:
        x = torch.rand(8, MAX_FEATURES)
        output = self.model(x)
        self.assertEqual(output.shape, (8, OUTPUT_DIM))

    def test_predict_returns_ndarray(self) -> None:
        x = torch.rand(5, MAX_FEATURES)
        result = predict(self.model, x, self.device)
        self.assertIsInstance(result, np.ndarray)

    def test_predict_output_is_binary(self) -> None:
        x = torch.rand(10, MAX_FEATURES)
        result = predict(self.model, x, self.device)
        self.assertTrue(set(result.tolist()).issubset({0, 1}))

    def test_predict_length_matches_input(self) -> None:
        x = torch.rand(7, MAX_FEATURES)
        result = predict(self.model, x, self.device)
        self.assertEqual(len(result), 7)


class TestSaveLoadModel(unittest.TestCase):
    def test_save_and_load(self) -> None:
        model = build_model(MAX_FEATURES, HIDDEN_DIM, OUTPUT_DIM)
        model.eval()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_model(model, path)
            loaded = load_model(path, MAX_FEATURES, HIDDEN_DIM, OUTPUT_DIM)
            x = torch.rand(3, MAX_FEATURES)
            with torch.no_grad():
                torch.testing.assert_close(model(x), loaded(x))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
