"""Model evaluation: metrics computation and reporting."""

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Compute evaluation metrics for the classifier.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, f1_score, precision, and recall.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def print_report(metrics: dict[str, float]) -> None:
    """Display evaluation metrics in a human-readable format.

    Args:
        metrics: Dictionary returned by evaluate_model.
    """
    for name, value in metrics.items():
        print(f"{name:<12}: {value:.4f}")
