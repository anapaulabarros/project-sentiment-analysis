"""Central entry point for the sentiment analysis pipeline."""

import numpy as np

from src.data.loader import load_data
from src.evaluation.metrics import print_report
from src.preprocessing.transform import preprocess_dataset
from src.training.train import run_training


def main() -> None:
    """Run the full sentiment analysis pipeline.

    Orchestrates data loading, preprocessing, dataset splitting,
    model training, and evaluation. Reports NumPy-based statistics
    throughout execution.
    """
    # Step 1: Load data
    print("Loading dataset...")
    df = load_data("data/raw/reviews.csv")
    print(f"  Raw dataset loaded: {len(df)} rows")

    # Step 2: Preprocess
    print("Preprocessing dataset...")
    df = preprocess_dataset(df)

    # NumPy statistics on preprocessed data (Requirement 15.1)
    data_array = np.asarray(df.values)
    print(f"  Preprocessed dataset shape (NumPy): {data_array.shape}")
    print(f"  Rows after preprocessing: {len(df)}")

    # Sentiment distribution using NumPy
    sentiment_arr = np.asarray(df["sentiment"], dtype=int)
    print(f"  Positive samples (np.sum): {np.sum(sentiment_arr == 1)}")
    print(f"  Negative samples (np.sum): {np.sum(sentiment_arr == 0)}")
    print(f"  Positive ratio (np.mean): {np.mean(sentiment_arr):.4f}")

    # Step 3: Train and evaluate
    print("\nTraining model...")
    results = run_training(df)

    # Step 4: NumPy statistics on predictions (Requirements 15.2, 15.3)
    y_pred = np.asarray(results["predictions"], dtype=int)
    print(f"\n  Predictions shape (NumPy): {y_pred.shape}")
    print(f"  Predicted positive (np.sum): {np.sum(y_pred == 1)}")
    print(f"  Predicted negative (np.sum): {np.sum(y_pred == 0)}")
    print(f"  Positive ratio (np.mean): {np.mean(y_pred):.4f}")

    # Step 5: Report metrics
    print_report(results["metrics"])


if __name__ == "__main__":
    main()
