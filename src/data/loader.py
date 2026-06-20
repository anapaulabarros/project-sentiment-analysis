"""Data loading and validation for the product reviews dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import RATING_COLUMN, TEXT_COLUMN


def load_data(path: str) -> pd.DataFrame:
    """Load the product reviews dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing the loaded reviews.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If required columns are missing from the dataset.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {path}")
    validate_columns(df, [TEXT_COLUMN, RATING_COLUMN])
    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Verify that all required columns are present in the DataFrame.

    Args:
        df: DataFrame to validate.
        required: List of required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def inspect_data(df: pd.DataFrame) -> None:
    """Print a NumPy-based summary of the dataset dimensions and rating distribution.

    Args:
        df: Loaded DataFrame.
    """
    ratings: np.ndarray = df[RATING_COLUMN].dropna().to_numpy()
    print(f"Total samples  : {df.shape[0]}")
    print(f"Total columns  : {df.shape[1]}")
    print(f"Rating min/max : {np.min(ratings):.0f} / {np.max(ratings):.0f}")
    print(f"Rating mean    : {np.mean(ratings):.2f}")
    print(f"Rating std     : {np.std(ratings):.2f}")
    unique, counts = np.unique(ratings.astype(int), return_counts=True)
    for rating, count in zip(unique, counts):
        print(f"  Rating {rating}: {count} reviews")
