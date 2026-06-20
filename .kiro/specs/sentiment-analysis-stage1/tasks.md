# Implementation Plan: Sentiment Analysis Stage 1

## Overview

Implement the full binary sentiment classification pipeline for Amazon product reviews. All modules exist with stub functions — this plan fills in the implementations with heavy NumPy usage, scikit-learn Pipeline for model construction, and comprehensive testing with pytest and hypothesis.

## Tasks

- [ ] 1. Implement core data loading and configuration
  - [ ] 1.1 Implement `src/data/loader.py` (load_data and validate_columns)
    - Implement `load_data`: read CSV with pandas, raise FileNotFoundError if path missing, call validate_columns
    - Implement `validate_columns`: check required columns exist, raise ValueError with missing names
    - Import configuration constants from `src/utils/config.py`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 16.4_

  - [ ]* 1.2 Write unit tests for data loader
    - Implement all tests in `tests/test_data_loader.py` with real assertions
    - Use synthetic CSV via tmp_path fixture
    - Test FileNotFoundError, ValueError for missing columns, successful load
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 2. Implement text preprocessing and label normalization
  - [ ] 2.1 Implement `src/preprocessing/transform.py` (clean_text, normalize_label, preprocess_dataset)
    - Implement `clean_text`: lowercase, remove punctuation via str.translate, collapse whitespace with re.sub
    - Implement `normalize_label`: rating >= 4 → 1, rating <= 2 → 0, rating == 3 → None
    - Implement `preprocess_dataset`: drop NaN, apply clean_text, apply normalize_label, drop neutral/empty rows
    - Import constants from config
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 16.4_

  - [ ]* 2.2 Write unit tests for preprocessing
    - Implement all tests in `tests/test_preprocessing.py` with real assertions
    - Test lowercase conversion, punctuation removal, whitespace collapsing
    - Test normalize_label for all rating values
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3_

  - [ ]* 2.3 Write property tests for text cleaning (Property 1 and 2)
    - **Property 1: Text cleaning idempotency** — clean_text(clean_text(x)) == clean_text(x)
    - **Property 2: Clean text output format** — result is lowercase, no punctuation, no double spaces
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**

  - [ ]* 2.4 Write property tests for label normalization (Property 3)
    - **Property 3: Label determinism and completeness** — normalize_label returns same value for same input, value in {0, 1, None}
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [ ] 3. Implement model construction and prediction
  - [ ] 3.1 Implement `src/models/model.py` (build_model, train_model, predict)
    - Implement `build_model`: create sklearn Pipeline with TfidfVectorizer(max_features=10000, ngram_range=(1,2)) + LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    - Implement `train_model`: call build_model, fit on data, return fitted pipeline
    - Implement `predict`: call model.predict, convert to np.asarray(predictions, dtype=int)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 8.1, 8.2, 8.3, 8.4, 16.4_

  - [ ]* 3.2 Write unit tests for model prediction
    - Implement all tests in `tests/test_predict.py` with real assertions
    - Train on small synthetic data, verify predict returns np.ndarray of {0, 1}
    - Verify output length matches input length
    - _Requirements: 8.1, 8.2, 8.3_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement training orchestration with NumPy split
  - [ ] 5.1 Implement `src/training/train.py` (split_dataset and run_training)
    - Implement `split_dataset`: use np.random.default_rng(RANDOM_SEED), np.arange, rng.shuffle, array slicing for train/test partition
    - Implement `run_training`: call split_dataset, train_model, predict, evaluate_model; return results dict
    - Raise ValueError if data has fewer than 2 rows
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 11.1, 11.2, 11.3, 16.4_

  - [ ]* 5.2 Write property tests for split conservation (Property 6)
    - **Property 6: Split conservation** — len(x_train) + len(x_test) == len(data), no rows created or lost
    - **Validates: Requirements 5.4**

  - [ ]* 5.3 Write property tests for split disjointness and reproducibility (Properties 7, 8, 9)
    - **Property 7: Split disjointness** — train and test share no data
    - **Property 8: Split size correctness** — test set size equals int(n * test_size)
    - **Property 9: Split reproducibility** — same seed produces identical splits
    - **Validates: Requirements 5.3, 5.5, 5.6**

- [ ] 6. Implement evaluation metrics with NumPy
  - [ ] 6.1 Implement `src/evaluation/metrics.py` (evaluate_model and print_report)
    - Implement `evaluate_model`: use np.asarray, boolean indexing for TP/TN/FP/FN, np.sum for counts, compute accuracy/precision/recall/F1
    - Implement `print_report`: format and print each metric with 4 decimal places
    - Do NOT use sklearn.metrics — compute everything with NumPy
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 10.1, 10.2, 16.4_

  - [ ]* 6.2 Write property test for metrics formula correctness (Property 11)
    - **Property 11: Metrics formula correctness** — for any binary arrays, all metrics in [0.0, 1.0], division-by-zero returns 0.0
    - **Validates: Requirements 9.4, 9.5, 9.6, 9.7, 9.8**

- [ ] 7. Implement main pipeline entry point
  - [ ] 7.1 Implement `main.py` with full pipeline orchestration and NumPy statistics
    - Call load_data, preprocess_dataset, run_training, print_report in sequence
    - After preprocessing: report dataset shape using np.asarray conversion
    - After predictions: compute np.mean, np.sum for positive/negative counts
    - Print NumPy statistics (positive ratio, prediction counts, dataset dimensions)
    - _Requirements: 12.1, 12.2, 12.3, 15.1, 15.2, 15.3_

- [ ] 8. Finalize dependencies and run all tests
  - [ ] 8.1 Update `requirements.txt` with pytest and hypothesis
    - Add pytest>=7.0 and hypothesis>=6.0 to requirements.txt
    - _Requirements: 14.1_

  - [ ]* 8.2 Write property test for prediction output invariant (Property 10)
    - **Property 10: Prediction output invariant** — for any n inputs, predict returns np.ndarray of shape (n,) with values in {0, 1}
    - **Validates: Requirements 8.1, 8.2, 8.3**

  - [ ]* 8.3 Write property tests for preprocessing (Properties 4 and 5)
    - **Property 4: Preprocessing row reduction** — output has no more rows than input
    - **Property 5: Preprocessing output validity** — all text satisfies clean_text postconditions, all sentiment in {0, 1}, no NaN
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6, 4.7**

- [ ] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All numerical operations use NumPy explicitly (course deliverable #4)
- sklearn.metrics is NOT used — all evaluation is NumPy-based
- sklearn.model_selection.train_test_split is NOT used — split uses NumPy shuffle

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "2.3", "2.4", "3.1", "6.1"] },
    { "id": 2, "tasks": ["3.2", "5.1", "6.2"] },
    { "id": 3, "tasks": ["5.2", "5.3", "7.1", "8.1"] },
    { "id": 4, "tasks": ["8.2", "8.3"] }
  ]
}
```
