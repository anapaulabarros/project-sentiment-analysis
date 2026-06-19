"""Training pipeline: dataset splitting and model training orchestration."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_SEED,
)
from src.models.model import train_model


def split_dataset(
    data: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into training and test sets.

    Args:
        data: Preprocessed DataFrame.
        text_column: Name of the text column.
        label_column: Name of the label column.
        test_size: Proportion of the dataset to use for testing (0–1).

    Returns:
        Tuple of (x_train, x_test, y_train, y_test).
    """
    x = data[text_column]
    y = data[label_column]
    return train_test_split(x, y, test_size=test_size, random_state=RANDOM_SEED)


def run_training(data: pd.DataFrame) -> object:
    """Execute the full training pipeline.

    Args:
        data: Preprocessed DataFrame.

    Returns:
        Trained classification model.
    """
    x_train, _x_test, y_train, _y_test = split_dataset(
        data,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        test_size=TEST_SIZE,
    )
    return train_model(x_train, y_train)
