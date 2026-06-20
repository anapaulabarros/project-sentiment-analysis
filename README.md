# Sentiment Analysis — Product Reviews

An Artificial Intelligence application for **binary sentiment classification** of product reviews. Given a review text, the system predicts whether the sentiment is **positive** or **negative**.

## Problem

Customer reviews on e-commerce platforms contain valuable information, but their volume makes manual reading unfeasible. This project builds an NLP pipeline to automatically classify the sentiment of textual reviews, handling noise, informal language, and linguistic variations.

Classification is **binary**:
- **Positive (1)**: rating ≥ 4 stars
- **Negative (0)**: rating ≤ 2 stars
- Rating 3 (neutral/ambiguous) is discarded

## Dataset

**Amazon Product Reviews** — [Kaggle: yasserh/amazon-product-reviews-dataset](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset)

Columns used:
- `reviews.text`: review text
- `reviews.rating`: rating from 1 to 5

### Downloading the dataset

```bash
# Via Kaggle CLI (requires kaggle.json configured)
kaggle datasets download -d yasserh/amazon-product-reviews-dataset
unzip amazon-product-reviews-dataset.zip -d data/raw/
mv data/raw/*.csv data/raw/reviews.csv
```

Or download manually from the Kaggle website and place the CSV at `data/raw/reviews.csv`.

## Project Structure

```
project-sentiment-analysis/
├── main.py                        # Central pipeline entry point
├── requirements.txt
├── LICENSE
├── data/
│   ├── raw/                       # Original dataset (not versioned)
│   └── processed/                 # Preprocessed data (generated)
├── docs/
│   ├── requirements.md            # Functional and non-functional requirements
│   ├── architecture.md            # System architecture and data flow
│   └── vision.md                  # System vision document
├── notebooks/
│   ├── sentiment_analysis.ipynb   # Interactive pipeline walkthrough
│   └── experiments.ipynb          # Hyperparameter comparison experiments
├── results/
│   ├── metrics/                   # metrics.json (generated)
│   ├── figures/                   # confusion_matrix.png, experiment_comparison.png (generated)
│   └── models/                    # model.pt, vectorizer.pkl (generated)
├── src/
│   ├── data/
│   │   └── loader.py              # Dataset loading, validation, NumPy inspection
│   ├── preprocessing/
│   │   └── transform.py           # Text cleaning, label normalization, TF-IDF + NumPy
│   ├── models/
│   │   └── model.py               # SentimentMLP (PyTorch nn.Module), save/load
│   ├── training/
│   │   └── train.py               # Split, tensors, DataLoader, train/val loops
│   ├── evaluation/
│   │   └── metrics.py             # Accuracy, F1, Precision, Recall
│   ├── inference/
│   │   └── predict.py             # Single and batch inference
│   └── utils/
│       └── config.py              # Global constants and hyperparameters
└── tests/
    ├── test_data.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_training.py
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd project-sentiment-analysis

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/sentiment_analysis.ipynb` | Full pipeline walkthrough organized by delivery stage (Entregas 1–8 + Final) |
| `notebooks/experiments.ipynb` | Hyperparameter comparison across 4 configurations (hidden_dim, learning rate, epochs) with results table and bar chart |

## NLP Pipeline

```
data/raw/reviews.csv
        │
        ▼
load_data()            # pandas CSV load + NumPy statistics
        │
        ▼
preprocess_dataset()   # lowercase, remove punctuation, rating → 0/1
        │
        ▼
vectorize_texts()      # TF-IDF → dense NumPy matrix
normalize_features()   # L2 normalization with np.linalg.norm
        │
        ▼
to_tensors()           # NumPy → PyTorch tensors
create_dataloader()    # TensorDataset + DataLoader
        │
        ▼
SentimentMLP           # nn.Module: Linear → ReLU → Dropout → Linear
run_training()         # BCEWithLogitsLoss + Adam, train + val loop
        │
        ▼
predict: 0 (NEGATIVE) | 1 (POSITIVE)
```

## Current Model

| Component  | Choice                              |
|------------|-------------------------------------|
| Features   | TF-IDF (max_features=2000, NumPy)   |
| Classifier | MLP PyTorch (hidden_dim=256)        |
| Loss       | BCEWithLogitsLoss                   |
| Optimizer  | Adam (lr=1e-3)                      |
| Evaluation | Accuracy, F1, Precision, Recall     |

## Documentation (`docs/`)

| File | Content |
|------|---------|
| `docs/vision.md` | System vision: problem, target audience, justification, scope, limitations, and potential impacts |
| `docs/requirements.md` | Functional requirements (RF01–RF08), non-functional requirements (RNF01–RNF08), constraints, and acceptance criteria |
| `docs/architecture.md` | Layer diagram, module responsibilities, data flow, and design decisions |

## Experimental Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9384 |
| F1-score  | 0.9682 |
| Precision | 0.9384 |
| Recall    | 1.0000 |

## Git Workflow

Development follows a feature-branch strategy:

```
main                  ← stable, protected
└── feature/clayton   ← active development branch
```

Each delivery stage was implemented incrementally with descriptive commit messages. To view the full history:

```bash
git log --oneline --graph feature/clayton
```

## Testing

```bash
python -m unittest discover tests -v
```

Expected output:

```
test_load_data_raises_file_not_found (tests.test_data.TestLoadData) ... ok
test_load_data_returns_dataframe (tests.test_data.TestLoadData) ... ok
test_passes_with_required_columns (tests.test_data.TestValidateColumns) ... ok
test_raises_on_missing_column (tests.test_data.TestValidateColumns) ... ok
test_forward_output_shape (tests.test_model.TestSentimentMLP) ... ok
test_predict_length_matches_input (tests.test_model.TestSentimentMLP) ... ok
test_predict_output_is_binary (tests.test_model.TestSentimentMLP) ... ok
test_predict_returns_ndarray (tests.test_model.TestSentimentMLP) ... ok
test_save_and_load (tests.test_model.TestSaveLoadModel) ... ok
test_converts_to_lowercase (tests.test_preprocessing.TestCleanText) ... ok
test_removes_extra_spaces (tests.test_preprocessing.TestCleanText) ... ok
test_removes_punctuation (tests.test_preprocessing.TestCleanText) ... ok
test_negative_rating (tests.test_preprocessing.TestNormalizeLabel) ... ok
test_neutral_rating (tests.test_preprocessing.TestNormalizeLabel) ... ok
test_positive_rating (tests.test_preprocessing.TestNormalizeLabel) ... ok
test_output_shape (tests.test_preprocessing.TestNormalizeFeatures) ... ok
test_unit_norms (tests.test_preprocessing.TestNormalizeFeatures) ... ok
test_shapes_preserved (tests.test_training.TestToTensors) ... ok
test_returns_float32_tensors (tests.test_training.TestToTensors) ... ok
test_batch_size (tests.test_training.TestCreateDataLoader) ... ok
test_dataset_length (tests.test_training.TestCreateDataLoader) ... ok
test_loss_is_positive (tests.test_training.TestTrainEpoch) ... ok
test_returns_float (tests.test_training.TestTrainEpoch) ... ok
test_loss_is_positive (tests.test_training.TestValidateEpoch) ... ok
test_returns_float (tests.test_training.TestValidateEpoch) ... ok
test_returns_sentiment_mlp (tests.test_training.TestRunTraining) ... ok

----------------------------------------------------------------------
Ran 26 tests in Xs

OK
```
