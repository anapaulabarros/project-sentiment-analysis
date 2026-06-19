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
