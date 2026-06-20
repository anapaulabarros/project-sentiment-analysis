"""Sentiment classification model definition."""

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.config import RANDOM_SEED


def build_model() -> Pipeline:
    """Instantiate the sentiment classification model.

    Returns:
        Initialized (unfitted) sklearn Pipeline with TfidfVectorizer and LogisticRegression.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])


def train_model(x_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """Train the classification model on the provided data.

    Args:
        x_train: Training features (text strings).
        y_train: Training labels (0 for negative, 1 for positive).

    Returns:
        Trained classification model (fitted Pipeline).
    """
    model = build_model()
    model.fit(x_train, y_train)
    return model


def predict(model: Pipeline, x: pd.Series) -> np.ndarray:
    """Generate sentiment predictions using the trained model.

    Args:
        model: Trained classification model.
        x: Input features (text strings).

    Returns:
        Array of predicted labels (0 or 1).
    """
    predictions = model.predict(x)
    return np.asarray(predictions, dtype=int)
