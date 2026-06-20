"""Training pipeline: dataset splitting and model training orchestration."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.models.model import predict, train_model
from src.utils.config import LABEL_COLUMN, RANDOM_SEED, TEST_SIZE, TEXT_COLUMN


def split_dataset(
    data: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into training and test sets using NumPy-based shuffling.

    Args:
        data: Preprocessed DataFrame.
        text_column: Name of the text column.
        label_column: Name of the label column.
        test_size: Proportion of the dataset to use for testing (0–1).

    Returns:
        Tuple of (x_train, x_test, y_train, y_test).
    """
    # Step 1: Create reproducible RNG
    rng = np.random.default_rng(RANDOM_SEED)

    # Step 2: Generate shuffled indices
    n = len(data)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Step 3: Compute split point
    n_test = int(n * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Step 4: Split data using NumPy array slicing
    x_train = data[text_column].iloc[train_indices].reset_index(drop=True)
    x_test = data[text_column].iloc[test_indices].reset_index(drop=True)
    y_train = data[label_column].iloc[train_indices].reset_index(drop=True)
    y_test = data[label_column].iloc[test_indices].reset_index(drop=True)

    return x_train, x_test, y_train, y_test


def run_training(data: pd.DataFrame) -> dict[str, object]:
    """Execute the full training pipeline.

    Args:
        data: Preprocessed DataFrame with text and sentiment columns.

    Returns:
        Dictionary with keys: model, predictions, y_test, metrics.

    Raises:
        ValueError: If data has fewer than 2 rows.
    """
    if len(data) < 2:
        raise ValueError("Dataset must have at least 2 rows for training.")

    # Split
    x_train, x_test, y_train, y_test = split_dataset(
        data, TEXT_COLUMN, LABEL_COLUMN, TEST_SIZE
    )

    # Train
    model = train_model(x_train, y_train)

    # Predict
    y_pred = predict(model, x_test)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred)

    return {
        "model": model,
        "predictions": y_pred,
        "y_test": y_test,
        "metrics": metrics,
    }
