"""Inference utilities: load artifacts and predict sentiment for new texts."""

import pickle

import numpy as np
import torch

from src.models.model import SentimentMLP, load_model
from src.preprocessing.transform import clean_text, normalize_features
from src.utils.config import (
    HIDDEN_DIM,
    MAX_FEATURES,
    MODEL_PATH,
    OUTPUT_DIM,
    VECTORIZER_PATH,
)


def load_artifacts(
    model_path: str = MODEL_PATH,
    vectorizer_path: str = VECTORIZER_PATH,
) -> tuple:
    """Load the trained model and TF-IDF vectorizer from disk.

    Args:
        model_path: Path to the saved model weights (.pt).
        vectorizer_path: Path to the saved vectorizer (.pkl).

    Returns:
        Tuple of (SentimentMLP in eval mode, fitted TfidfVectorizer).
    """
    model = load_model(model_path, MAX_FEATURES, HIDDEN_DIM, OUTPUT_DIM)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_single(
    text: str,
    model: SentimentMLP,
    vectorizer,
    device: torch.device,
) -> int:
    """Predict the sentiment of a single review text.

    Args:
        text: Raw review text.
        model: Trained SentimentMLP.
        vectorizer: Fitted TfidfVectorizer.
        device: Target device.

    Returns:
        1 for positive sentiment, 0 for negative.
    """
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned]).toarray()
    X = normalize_features(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(X_tensor).squeeze()
        label = int(torch.sigmoid(logit).item() >= 0.5)
    return label


def predict_batch(
    texts: list[str],
    model: SentimentMLP,
    vectorizer,
    device: torch.device,
) -> np.ndarray:
    """Predict sentiment for a list of review texts.

    Args:
        texts: List of raw review texts.
        model: Trained SentimentMLP.
        vectorizer: Fitted TfidfVectorizer.
        device: Target device.

    Returns:
        NumPy array of binary labels (0 or 1).
    """
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned).toarray()
    X = normalize_features(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(logits)
        labels = (probs >= 0.5).long()
    return labels.cpu().numpy()
