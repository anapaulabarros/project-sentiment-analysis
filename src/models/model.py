"""Sentiment classification model definition."""

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_model() -> object:
    """Instantiate the sentiment classification model.

    Returns:
        Initialized classification model.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def train_model(x_train: Sequence, y_train: Sequence[int]) -> object:
    """Train the classification model on the provided data.

    Args:
        x_train: Training features.
        y_train: Training labels (0 for negative, 1 for positive).

    Returns:
        Trained classification model.
    """
    model = build_model()
    model.fit(x_train, y_train)
    return model


def predict(model: object, x: Sequence) -> np.ndarray:
    """Generate sentiment predictions using the trained model.

    Args:
        model: Trained classification model.
        x: Input features.

    Returns:
        Array of predicted labels (0 or 1).
    """
    return model.predict(x)
