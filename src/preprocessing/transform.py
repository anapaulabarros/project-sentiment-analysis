"""Text preprocessing and label normalization transformations."""

import pandas as pd


def clean_text(text: str) -> str:
    """Remove noise and normalize a review text.

    Applies lowercasing, punctuation removal, and whitespace collapsing.

    Args:
        text: Raw review text.

    Returns:
        Cleaned and normalized text.
    """
    pass


def normalize_label(rating: int) -> int | None:
    """Convert a numeric rating into a binary sentiment label.

    Args:
        rating: Review rating from 1 to 5.

    Returns:
        1 for positive sentiment, 0 for negative, None for neutral (discarded).
    """
    pass


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and label normalization to the full dataset.

    Drops rows with missing text, missing rating, or neutral rating (3).

    Args:
        df: DataFrame with text and rating columns.

    Returns:
        Processed DataFrame with a binary sentiment column added.
    """
    pass
