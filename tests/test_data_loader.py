"""Unit tests for the data loader module."""

import os
import tempfile

import pandas as pd
import pytest

from src.data.loader import load_data, validate_columns


def test_load_data_returns_dataframe() -> None:
    """load_data should return a DataFrame when given a valid CSV file."""
    sample = pd.DataFrame({
        "reviews.text": ["great product", "terrible quality"],
        "reviews.rating": [5, 1],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        sample.to_csv(f, index=False)
        tmp_path = f.name

    try:
        result = load_data(tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
    finally:
        os.unlink(tmp_path)


def test_load_data_raises_file_not_found() -> None:
    """load_data should raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


def test_validate_columns_passes_with_required_columns() -> None:
    """validate_columns should not raise when all required columns are present."""
    df = pd.DataFrame({"reviews.text": [], "reviews.rating": []})
    validate_columns(df, ["reviews.text", "reviews.rating"])  # should not raise


def test_validate_columns_raises_on_missing_column() -> None:
    """validate_columns should raise ValueError when a column is missing."""
    df = pd.DataFrame({"reviews.text": []})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(df, ["reviews.text", "reviews.rating"])
