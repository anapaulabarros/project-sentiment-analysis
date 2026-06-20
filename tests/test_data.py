"""Unit tests for the data loader module."""

import unittest

import pandas as pd

from src.data.loader import load_data, validate_columns
from src.utils.config import DATA_PATH


class TestLoadData(unittest.TestCase):
    def test_load_data_returns_dataframe(self) -> None:
        """load_data should return a DataFrame when given a valid CSV file."""
        df = load_data(DATA_PATH)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_load_data_raises_file_not_found(self) -> None:
        """load_data should raise FileNotFoundError for a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")


class TestValidateColumns(unittest.TestCase):
    def test_passes_with_required_columns(self) -> None:
        """validate_columns should not raise when all required columns are present."""
        df = pd.DataFrame({"reviews.text": ["good"], "reviews.rating": [5]})
        validate_columns(df, ["reviews.text", "reviews.rating"])

    def test_raises_on_missing_column(self) -> None:
        """validate_columns should raise ValueError when a column is missing."""
        df = pd.DataFrame({"reviews.text": ["good"]})
        with self.assertRaises(ValueError):
            validate_columns(df, ["reviews.text", "reviews.rating"])


if __name__ == "__main__":
    unittest.main()
