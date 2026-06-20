"""Unit tests for the preprocessing module."""

import unittest

import numpy as np

from src.preprocessing.transform import clean_text, normalize_features, normalize_label


class TestCleanText(unittest.TestCase):
    def test_converts_to_lowercase(self) -> None:
        self.assertEqual(clean_text("Hello World"), "hello world")

    def test_removes_extra_spaces(self) -> None:
        self.assertEqual(clean_text("hello   world"), "hello world")

    def test_removes_punctuation(self) -> None:
        self.assertEqual(clean_text("hello, world!"), "hello world")


class TestNormalizeLabel(unittest.TestCase):
    def test_positive_rating(self) -> None:
        self.assertEqual(normalize_label(4), 1)
        self.assertEqual(normalize_label(5), 1)

    def test_negative_rating(self) -> None:
        self.assertEqual(normalize_label(1), 0)
        self.assertEqual(normalize_label(2), 0)

    def test_neutral_rating(self) -> None:
        self.assertIsNone(normalize_label(3))


class TestNormalizeFeatures(unittest.TestCase):
    def test_output_shape(self) -> None:
        X = np.random.rand(10, 100)
        result = normalize_features(X)
        self.assertEqual(result.shape, X.shape)

    def test_unit_norms(self) -> None:
        X = np.random.rand(10, 100)
        result = normalize_features(X)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
