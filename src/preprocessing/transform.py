"""Text preprocessing and label normalization transformations."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import (
    LABEL_COLUMN,
    MAX_NEGATIVE_RATING,
    MIN_POSITIVE_RATING,
    RATING_COLUMN,
    TEXT_COLUMN,
)


def clean_text(text: str) -> str:
    """Remove noise and normalize a review text.

    Applies lowercasing, punctuation removal, and whitespace collapsing.

    Args:
        text: Raw review text.

    Returns:
        Cleaned and normalized text.
    """
    text = text.lower()
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
    return None


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and label normalization to the full dataset.

    Drops rows with missing text, missing rating, or neutral rating (3).

    Args:
        df: DataFrame with text and rating columns.

    Returns:
        Processed DataFrame with a binary sentiment column added.
    """
    df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(clean_text)
    df[LABEL_COLUMN] = df[RATING_COLUMN].astype(int).apply(normalize_label)
    df = df.dropna(subset=[LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df.reset_index(drop=True)


def vectorize_texts(
    texts: pd.Series,
    max_features: int,
) -> tuple[TfidfVectorizer, np.ndarray]:
    """Fit a TF-IDF vectorizer and convert texts to a dense NumPy matrix.

    Args:
        texts: Series of cleaned review texts.
        max_features: Maximum vocabulary size.

    Returns:
        Tuple of (fitted vectorizer, dense feature matrix).
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts).toarray()
    return vectorizer, X


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Apply L2 normalization to each sample using NumPy.

    Args:
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Row-normalized feature matrix.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms
