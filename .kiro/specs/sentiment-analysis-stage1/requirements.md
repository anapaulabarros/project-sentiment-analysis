# Requirements Document

## Introduction

Binary sentiment classification pipeline for Amazon product reviews covering course deliverables 1–4: focused functions, package-based modularization, type hints on all functions, and NumPy usage for data manipulation. The system loads CSV data, preprocesses text, normalizes labels, vectorizes text with NumPy Bag-of-Words, trains a LogisticRegression classifier, and evaluates with standard metrics.

## Glossary

- **Pipeline**: The end-to-end sequence of data loading, preprocessing, training, and evaluation
- **Loader**: The `src/data/loader.py` module responsible for CSV ingestion and schema validation
- **Preprocessor**: The `src/preprocessing/transform.py` module responsible for text cleaning, label normalization, vocabulary building, and vectorization
- **Trainer**: The `src/training/train.py` module responsible for dataset splitting, orchestrating training and evaluation
- **Model**: The `src/models/model.py` module responsible for LogisticRegression fitting and prediction
- **Evaluator**: The `src/evaluation/metrics.py` module responsible for computing classification metrics using NumPy
- **Vocabulary**: A dictionary mapping each unique word from training texts to a unique integer index
- **Count_Matrix**: A NumPy 2D array where each row is a document and each column is a word count
- **Sentiment_Label**: Binary integer value: 1 (positive) or 0 (negative)

## Requirements

### Requirement 1: Data Loading

**User Story:** As a data scientist, I want to load product review data from a CSV file, so that I can feed it into the analysis pipeline.

#### Acceptance Criteria

1. WHEN a valid CSV file path is provided, THE Loader SHALL return a pandas DataFrame containing the file contents
2. WHEN the file path does not exist, THE Loader SHALL raise a FileNotFoundError
3. WHEN required columns (`reviews.text`, `reviews.rating`) are absent from the CSV, THE Loader SHALL raise a ValueError listing the missing columns
4. WHEN the CSV file contains zero data rows, THE Loader SHALL raise a ValueError

### Requirement 2: Text Cleaning

**User Story:** As a data scientist, I want review text cleaned and normalized, so that the model receives consistent input.

#### Acceptance Criteria

1. WHEN text is cleaned, THE Preprocessor SHALL convert all characters to lowercase
2. WHEN text is cleaned, THE Preprocessor SHALL remove all punctuation characters
3. WHEN text is cleaned, THE Preprocessor SHALL collapse consecutive whitespace into a single space and strip leading/trailing whitespace
4. WHEN text is cleaned, THE Preprocessor SHALL retain only alphanumeric characters and spaces
5. THE Preprocessor SHALL produce the same output when `clean_text` is applied to already-cleaned text (idempotency)

### Requirement 3: Label Normalization

**User Story:** As a data scientist, I want star ratings converted to binary sentiment labels, so that I can train a binary classifier.

#### Acceptance Criteria

1. WHEN a rating is 4 or 5, THE Preprocessor SHALL return Sentiment_Label 1 (positive)
2. WHEN a rating is 1 or 2, THE Preprocessor SHALL return Sentiment_Label 0 (negative)
3. WHEN a rating is 3, THE Preprocessor SHALL return None (neutral, to be discarded)
4. THE Preprocessor SHALL produce a deterministic output for any given rating value

### Requirement 4: Dataset Preprocessing

**User Story:** As a data scientist, I want the full dataset cleaned and filtered in one step, so that it is ready for training.

#### Acceptance Criteria

1. WHEN the dataset is preprocessed, THE Preprocessor SHALL drop rows with missing text or missing rating values
2. WHEN the dataset is preprocessed, THE Preprocessor SHALL apply `clean_text` to every review text
3. WHEN the dataset is preprocessed, THE Preprocessor SHALL apply `normalize_label` to every rating
4. WHEN the dataset is preprocessed, THE Preprocessor SHALL drop rows where the label is None (neutral reviews)
5. WHEN the dataset is preprocessed, THE Preprocessor SHALL drop rows where the cleaned text is empty
6. THE Preprocessor SHALL return a DataFrame with row count less than or equal to the input row count

### Requirement 5: Vocabulary Building and Vectorization

**User Story:** As a data scientist, I want text converted to numerical features using NumPy, so that the model can process them.

#### Acceptance Criteria

1. WHEN vocabulary is built from training texts, THE Preprocessor SHALL assign a unique contiguous integer index (starting at 0) to each distinct word
2. WHEN vocabulary is built, THE Preprocessor SHALL include every distinct word present in the training texts
3. WHEN texts are vectorized, THE Preprocessor SHALL produce a Count_Matrix of shape (number_of_texts, vocabulary_size)
4. WHEN texts are vectorized, THE Preprocessor SHALL set each cell to the count of the corresponding word in the corresponding text
5. WHEN a word in a text is not in the Vocabulary, THE Preprocessor SHALL assign it a count of zero (ignore it)
6. THE Preprocessor SHALL produce Count_Matrix values that are all non-negative

### Requirement 6: Train/Test Split

**User Story:** As a data scientist, I want the dataset split into training and testing subsets using NumPy, so that I can evaluate model generalization.

#### Acceptance Criteria

1. WHEN the dataset is split, THE Trainer SHALL produce training and test sets whose combined length equals the original dataset length
2. WHEN the dataset is split, THE Trainer SHALL use NumPy random number generation for index shuffling
3. WHEN the dataset is split, THE Trainer SHALL use the configured RANDOM_SEED for reproducibility
4. WHEN split is called twice with the same seed and data, THE Trainer SHALL produce identical results
5. WHEN the dataset has fewer than 2 rows, THE Trainer SHALL raise a ValueError

### Requirement 7: Model Training

**User Story:** As a data scientist, I want a LogisticRegression model trained on the vectorized data, so that I can classify review sentiment.

#### Acceptance Criteria

1. WHEN training data is provided, THE Model SHALL fit a LogisticRegression classifier
2. WHEN the model is trained, THE Model SHALL use the configured RANDOM_SEED for reproducibility
3. THE Model SHALL accept text Series as input and handle vectorization internally via its pipeline

### Requirement 8: Prediction

**User Story:** As a data scientist, I want predictions from the trained model, so that I can evaluate its performance.

#### Acceptance Criteria

1. WHEN input texts are provided to a fitted model, THE Model SHALL return predictions as a NumPy array
2. THE Model SHALL produce predictions containing only values 0 or 1
3. THE Model SHALL return a prediction array with the same length as the input

### Requirement 9: Evaluation Metrics

**User Story:** As a data scientist, I want accuracy, F1, precision, and recall computed with NumPy, so that I can assess model quality.

#### Acceptance Criteria

1. WHEN true labels and predicted labels are provided, THE Evaluator SHALL compute accuracy using NumPy operations
2. WHEN true labels and predicted labels are provided, THE Evaluator SHALL compute precision using NumPy operations
3. WHEN true labels and predicted labels are provided, THE Evaluator SHALL compute recall using NumPy operations
4. WHEN true labels and predicted labels are provided, THE Evaluator SHALL compute F1 score using NumPy operations
5. THE Evaluator SHALL return all metric values in the range [0.0, 1.0]
6. IF a division by zero occurs during metric computation, THEN THE Evaluator SHALL return 0.0 for that metric
7. THE Evaluator SHALL return a dictionary with keys: accuracy, f1_score, precision, recall
8. WHEN a metrics report is printed, THE Evaluator SHALL display all four metrics in a formatted table

### Requirement 10: Package Structure

**User Story:** As a developer, I want code organized in clearly separated packages, so that the project is maintainable and meets modularization requirements.

#### Acceptance Criteria

1. THE Pipeline SHALL organize code into packages: `src/data`, `src/preprocessing`, `src/models`, `src/training`, `src/evaluation`, `src/utils`
2. THE Pipeline SHALL centralize execution in `main.py` at the project root
3. WHEN a package is imported, THE Pipeline SHALL expose its public functions through `__init__.py` files

### Requirement 11: Type Hints

**User Story:** As a developer, I want all functions annotated with type hints, so that the code is self-documenting and statically checkable.

#### Acceptance Criteria

1. THE Pipeline SHALL annotate every function with parameter type hints and return type hints
2. THE Pipeline SHALL use standard Python typing constructs (e.g., `list[str]`, `dict[str, int]`, `tuple[...]`, `Sequence[int]`)

### Requirement 12: Pipeline Orchestration

**User Story:** As a user, I want to run the entire pipeline with a single command, so that I get results without manual steps.

#### Acceptance Criteria

1. WHEN `main.py` is executed, THE Pipeline SHALL load data, preprocess it, train the model, and print evaluation metrics
2. IF any step in the pipeline fails, THEN THE Pipeline SHALL propagate the error with a descriptive message
