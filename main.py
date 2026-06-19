"""Central entry point for the sentiment analysis pipeline."""

from src.data.loader import load_data
from src.preprocessing.transform import preprocess_dataset
from src.training.train import split_dataset, run_training
from src.evaluation.metrics import evaluate_model, print_report
from src.models.model import predict
from src.utils.config import (
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
)


def main() -> None:
    """Run the full sentiment analysis pipeline.

    Orchestrates data loading, preprocessing, dataset splitting,
    model training, and evaluation.
    """
    dataset_path = "data/raw/reviews.csv"

    print("Loading data...")
    df = load_data(dataset_path)

    print("Preprocessing...")
    df = preprocess_dataset(df)
    print(f"  Samples after preprocessing: {len(df)}")

    print("Splitting dataset...")
    x_train, x_test, y_train, y_test = split_dataset(
        df,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        test_size=TEST_SIZE,
    )
    print(f"  Train: {len(x_train)} | Test: {len(x_test)}")

    print("Training model...")
    model = run_training(df)

    print("Evaluating...")
    y_pred = predict(model, x_test)
    metrics = evaluate_model(y_test, y_pred)
    print_report(metrics)


if __name__ == "__main__":
    main()
