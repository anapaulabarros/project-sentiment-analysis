"""Training pipeline: dataset splitting and model training orchestration."""

import pandas as pd


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
    pass


def run_training(data: pd.DataFrame) -> object:
    """Execute the full training pipeline.

    Args:
        data: Preprocessed DataFrame.

    Returns:
        Trained classification model.
    """
    pass
