"""Unit tests for the model prediction module."""

import numpy as np

from src.models.model import train_model, predict


# Minimal training data for testing purposes
_X_TRAIN = [
    "great product love it",
    "amazing quality highly recommend",
    "excellent very happy",
    "fantastic works perfectly",
    "terrible product broke immediately",
    "very bad quality waste of money",
    "awful do not buy",
    "horrible stopped working after one day",
]
_Y_TRAIN = [1, 1, 1, 1, 0, 0, 0, 0]


def test_predict_returns_array() -> None:
    """predict should return a numpy array of labels."""
    model = train_model(_X_TRAIN, _Y_TRAIN)
    result = predict(model, ["good product"])
    assert isinstance(result, np.ndarray)


def test_predict_output_is_binary() -> None:
    """predict should return only 0 or 1 as label values."""
    model = train_model(_X_TRAIN, _Y_TRAIN)
    result = predict(model, ["good product", "bad product"])
    assert set(result).issubset({0, 1})


def test_predict_batch_length_matches_input() -> None:
    """predict output length should match the number of input samples."""
    model = train_model(_X_TRAIN, _Y_TRAIN)
    inputs = ["good", "bad", "okay", "love it"]
    result = predict(model, inputs)
    assert len(result) == len(inputs)
