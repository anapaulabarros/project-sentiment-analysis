"""Unit tests for the data loader module."""

from src.data.loader import load_data, validate_columns


def test_load_data_returns_dataframe() -> None:
    """load_data should return a DataFrame when given a valid CSV file."""
    pass


def test_load_data_raises_file_not_found() -> None:
    """load_data should raise FileNotFoundError for a non-existent path."""
    pass


def test_validate_columns_passes_with_required_columns() -> None:
    """validate_columns should not raise when all required columns are present."""
    pass


def test_validate_columns_raises_on_missing_column() -> None:
    """validate_columns should raise ValueError when a column is missing."""
    pass
