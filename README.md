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
├── data/
│   ├── raw/                       # Original dataset (not versioned)
│   └── processed/                 # Preprocessed data
├── results/
│   ├── metrics/                   # Evaluation metrics per experiment
│   └── figures/                   # Plots and visualizations
└── src/
    ├── data/
    │   └── loader.py              # Dataset loading and validation
    ├── preprocessing/
    │   └── transform.py           # Text cleaning and label normalization
    ├── models/
    │   └── model.py               # Model definition and prediction
    ├── training/
    │   └── train.py               # Dataset splitting and training
    ├── evaluation/
    │   └── metrics.py             # Evaluation metrics and reporting
    └── utils/
        └── config.py              # Global constants and parameters
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

## NLP Pipeline

```
Raw text
    │
    ▼
clean_text()           # lowercase, remove punctuation, collapse spaces
    │
    ▼
normalize_label()      # rating → 0/1 (discard neutral)
    │
    ▼
Feature extraction     # numerical text representation
    │
    ▼
Classification model   # binary classifier
    │
    ▼
predict: 0 (NEGATIVE) | 1 (POSITIVE)
```

## Current Model

| Component     | Choice                          |
|---------------|---------------------------------|
| Features      | To be defined                   |
| Classifier    | To be defined                   |
| Evaluation    | Accuracy, F1, Precision, Recall |

Deep learning models with PyTorch will be introduced in future deliveries.
