"""Model evaluation: metrics computation and reporting."""

from typing import Sequence

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def print_report(metrics: dict[str, float]) -> None:
    """Display evaluation metrics in a human-readable format.

    Args:
        metrics: Dictionary returned by evaluate_model.
    """
    print("\n=== Evaluation Results ===")
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("==========================\n")
