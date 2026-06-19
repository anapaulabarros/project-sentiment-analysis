"""Data loading and validation for the product reviews dataset."""

import pandas as pd


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
    pass


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Verify that all required columns are present in the DataFrame.

    Args:
        df: DataFrame to validate.
        required: List of required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    pass
