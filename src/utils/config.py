"""Global constants and configuration parameters for the project."""

# Dataset columns (Amazon Product Reviews - Kaggle: yasserh)
TEXT_COLUMN: str = "reviews.text"
RATING_COLUMN: str = "reviews.rating"
LABEL_COLUMN: str = "sentiment"

# Sentiment derivation from rating
MIN_POSITIVE_RATING: int = 4  # rating >= 4 → positive (1)
MAX_NEGATIVE_RATING: int = 2  # rating <= 2 → negative (0)
# rating == 3 → neutral, discarded

# Train/test split
TEST_SIZE: float = 0.2
RANDOM_SEED: int = 42

# Paths
DATA_PATH: str = "data/raw/reviews.csv"
PROCESSED_PATH: str = "data/processed/reviews_processed.csv"
METRICS_PATH: str = "results/metrics/metrics.json"
FIGURES_PATH: str = "results/figures/confusion_matrix.png"

# PyTorch hyperparameters
MAX_FEATURES: int = 2000
HIDDEN_DIM: int = 256
OUTPUT_DIM: int = 1
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
BATCH_SIZE: int = 64

# Model artifacts
MODEL_PATH: str = "results/models/model.pt"
VECTORIZER_PATH: str = "results/models/vectorizer.pkl"
