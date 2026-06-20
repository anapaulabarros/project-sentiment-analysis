"""Unit tests for the model prediction module."""

import numpy as np
import pandas as pd

from src.models.model import build_model, predict, train_model


def _get_trained_model():
    """Helper: train a small model on synthetic data."""
    x_train = pd.Series([
        "this is great amazing wonderful",
        "love this product so much",
        "excellent quality highly recommend",
        "terrible awful worst ever",
        "bad product do not buy",
        "horrible waste of money",
    ])
    y_train = pd.Series([1, 1, 1, 0, 0, 0])
    return train_model(x_train, y_train)


def test_predict_returns_array() -> None:
    """predict should return a numpy array of labels."""
    model = _get_trained_model()
    x_test = pd.Series(["great product", "terrible quality"])
    result = predict(model, x_test)
    assert isinstance(result, np.ndarray)


def test_predict_output_is_binary() -> None:
    """predict should return only 0 or 1 as label values."""
    model = _get_trained_model()
    x_test = pd.Series(["great product", "terrible quality", "okay item"])
    result = predict(model, x_test)
    assert all(val in (0, 1) for val in result)


def test_predict_batch_length_matches_input() -> None:
    """predict output length should match the number of input samples."""
    model = _get_trained_model()
    x_test = pd.Series(["great", "bad", "okay", "wonderful"])
    result = predict(model, x_test)
    assert len(result) == len(x_test)
    assert result.shape == (4,)


def test_predict_dtype_is_int() -> None:
    """predict should return array with integer dtype."""
    model = _get_trained_model()
    x_test = pd.Series(["good product"])
    result = predict(model, x_test)
    assert np.issubdtype(result.dtype, np.integer)


def test_build_model_returns_unfitted_pipeline() -> None:
    """build_model should return a Pipeline with tfidf and clf steps."""
    model = build_model()
    assert hasattr(model, "steps")
    step_names = [name for name, _ in model.steps]
    assert "tfidf" in step_names
    assert "clf" in step_names
