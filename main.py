"""Central entry point for the sentiment analysis pipeline."""

from src.data.loader import load_data
from src.preprocessing.transform import preprocess_dataset
from src.training.train import split_dataset, run_training
from src.evaluation.metrics import evaluate_model, print_report


def main() -> None:
    """Run the full sentiment analysis pipeline.

    Orchestrates data loading, preprocessing, dataset splitting,
    model training, and evaluation.
    """
    pass


if __name__ == "__main__":
    main()
