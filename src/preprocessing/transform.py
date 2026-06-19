"""Text preprocessing and label normalization transformations."""

import re

import pandas as pd

from src.utils.config import (
    TEXT_COLUMN,
    RATING_COLUMN,
    LABEL_COLUMN,
    MIN_POSITIVE_RATING,
    MAX_NEGATIVE_RATING,
)


def clean_text(text: str) -> str:
    """Remove noise and normalize a review text.

    Applies lowercasing, punctuation removal, and whitespace collapsing.

    Args:
        text: Raw review text.

    Returns:
        Cleaned and normalized text.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(rating: int) -> int | None:
    """Convert a numeric rating into a binary sentiment label.

    Args:
        rating: Review rating from 1 to 5.

    Returns:
        1 for positive sentiment, 0 for negative, None for neutral (discarded).
    """
    if rating >= MIN_POSITIVE_RATING:
        return 1
    if rating <= MAX_NEGATIVE_RATING:
        return 0
    return None  # neutral rating (3) — discarded


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and label normalization to the full dataset.

    Drops rows with missing text, missing rating, or neutral rating (3).

    Args:
        df: DataFrame with text and rating columns.

    Returns:
        Processed DataFrame with a binary sentiment column added.
    """
    df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).copy()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
    df[LABEL_COLUMN] = df[RATING_COLUMN].apply(normalize_label)

    # Drop neutral rows (label == None)
    df = df.dropna(subset=[LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # Drop empty texts after cleaning
    df = df[df[TEXT_COLUMN].str.strip() != ""]

    return df.reset_index(drop=True)
