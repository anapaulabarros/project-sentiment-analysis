"""Model evaluation: metrics computation and reporting."""

from typing import Sequence


def evaluate_model(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Compute evaluation metrics for the classifier.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, f1_score, precision, and recall.
    """
    pass


def print_report(metrics: dict[str, float]) -> None:
    """Display evaluation metrics in a human-readable format.

    Args:
        metrics: Dictionary returned by evaluate_model.
    """
    pass
