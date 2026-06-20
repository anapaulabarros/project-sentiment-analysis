"""Central entry point for the sentiment analysis pipeline."""

import json
import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data.loader import inspect_data, load_data
from src.evaluation.metrics import evaluate_model, print_report
from src.models.model import build_model, predict, save_model
from src.preprocessing.transform import (
    normalize_features,
    preprocess_dataset,
    vectorize_texts,
)
from src.training.train import (
    create_dataloader,
    run_training,
    split_dataset,
    to_tensors,
)
from src.utils.config import (
    BATCH_SIZE,
    DATA_PATH,
    FIGURES_PATH,
    HIDDEN_DIM,
    LABEL_COLUMN,
    LEARNING_RATE,
    MAX_FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUM_EPOCHS,
    OUTPUT_DIM,
    PROCESSED_PATH,
    TEST_SIZE,
    TEXT_COLUMN,
    VECTORIZER_PATH,
)


def main() -> None:
    """Run the full sentiment analysis pipeline."""
    # 1. Load and preprocess
    df = load_data(DATA_PATH)
    inspect_data(df)
    df = preprocess_dataset(df)
    df.to_csv(PROCESSED_PATH, index=False)

    # 2. Split
    x_train, x_test, y_train, y_test = split_dataset(
        df, TEXT_COLUMN, LABEL_COLUMN, TEST_SIZE
    )

    # 3. Vectorize (NumPy)
    vectorizer, X_train_np = vectorize_texts(x_train, MAX_FEATURES)
    X_test_np = vectorizer.transform(x_test).toarray()

    # 4. Normalize (NumPy)
    X_train_np = normalize_features(X_train_np)
    X_test_np = normalize_features(X_test_np)

    # 5. Convert to tensors and build DataLoaders
    X_train_t, y_train_t = to_tensors(X_train_np, y_train.to_numpy())
    X_test_t, y_test_t = to_tensors(X_test_np, y_test.to_numpy())

    train_loader = create_dataloader(X_train_t, y_train_t, BATCH_SIZE)
    val_loader = create_dataloader(X_test_t, y_test_t, BATCH_SIZE, shuffle=False)

    # 6. Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(MAX_FEATURES, HIDDEN_DIM, OUTPUT_DIM).to(device)

    # 7. Train
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    model = run_training(
        model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device
    )

    # 8. Evaluate
    y_pred = predict(model, X_test_t, device)
    metrics = evaluate_model(y_test.to_numpy(), y_pred)
    print_report(metrics)

    # 9. Save model and vectorizer
    save_model(model, MODEL_PATH)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    # 10. Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # 11. Save confusion matrix
    cm = confusion_matrix(y_test.to_numpy(), y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    )
    disp.plot(colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH)
    plt.close()


if __name__ == "__main__":
    main()
