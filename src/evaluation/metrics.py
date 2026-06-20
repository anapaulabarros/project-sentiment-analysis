"""Model evaluation: metrics computation and reporting."""

from typing import Sequence

import numpy as np


def evaluate_model(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Compute evaluation metrics for the classifier using NumPy.

    Uses NumPy boolean indexing and np.sum for all calculations.
    Does NOT use sklearn.metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, f1_score, precision, and recall.
    """
    # Convert to NumPy arrays explicitly
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    # Confusion matrix components using NumPy boolean indexing
    tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    tn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))

    # Compute metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
    }


def print_report(metrics: dict[str, float]) -> None:
    """Display evaluation metrics in a human-readable format.

    Args:
        metrics: Dictionary returned by evaluate_model.
    """
    print("\n" + "=" * 40)
    print("  Classification Metrics Report")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("=" * 40 + "\n")
