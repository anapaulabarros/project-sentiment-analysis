# Implementation Plan: Sentiment Analysis Stage 1

## Overview

Implement a binary sentiment classification pipeline for Amazon product reviews using focused functions, package structure, type hints, and NumPy-based operations. Each task implements one module's functions along with their tests. Some modules (loader, metrics) are already implemented and only need test verification.

## Tasks

- [x] 1. Implement data loader and tests
  - [x] 1.1 Verify and finalize `src/data/loader.py` implementation
    - `load_data` and `validate_columns` are already implemented
    - Verify they match design spec (FileNotFoundError, ValueError for missing columns, ValueError for empty CSV)
    - Ensure type hints are complete
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x]* 1.2 Write property tests for data loader
    - **Property: validate_columns raises ValueError if and only if a required column is missing**
    - **Validates: Requirements 1.3**
    - Use Hypothesis to generate DataFrames with random column subsets

- [x] 2. Implement text preprocessing and tests
  - [x] 2.1 Implement `build_vocabulary` and `texts_to_matrix` in `src/preprocessing/transform.py`
    - `clean_text`, `normalize_label`, and `preprocess_dataset` are already implemented
    - Implement `build_vocabulary(texts: pd.Series) -> dict[str, int]`: iterate words, assign contiguous indices
    - Implement `texts_to_matrix(texts: pd.Series, vocab: dict[str, int]) -> np.ndarray`: create NumPy count matrix
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x]* 2.2 Write property tests for text cleaning
    - **Property 1: Text cleaning idempotency** — `clean_text(clean_text(t)) == clean_text(t)`
    - **Property 2: Clean text output format** — result is lowercase, no punctuation, no leading/trailing whitespace, no consecutive spaces
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
    - Use Hypothesis `text()` strategy

  - [x]* 2.3 Write property tests for label normalization
    - **Property 3: Label determinism and correctness** — rating ≥ 4 → 1, rating ≤ 2 → 0, rating == 3 → None
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    - Use Hypothesis `integers(min_value=1, max_value=5)`

  - [x]* 2.4 Write property tests for vocabulary and vectorization
    - **Property 6: Vocabulary completeness and contiguity** — all words get unique contiguous indices
    - **Property 7: Matrix shape correctness** — shape == (len(texts), len(vocab))
    - **Property 8: Matrix non-negativity** — all values ≥ 0
    - **Property 9: Matrix count accuracy** — cell equals word occurrence count
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

- [x] 3. Implement train/test split and tests
  - [x] 3.1 Add validation to `split_dataset` in `src/training/train.py`
    - Add ValueError for datasets with fewer than 2 rows
    - Verify existing implementation matches design (NumPy shuffle, RANDOM_SEED)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x]* 3.2 Write property tests for split_dataset
    - **Property 10: Split conservation** — `len(x_train) + len(x_test) == len(data)`
    - **Property 11: Split reproducibility** — same seed produces identical splits
    - **Validates: Requirements 6.1, 6.3, 6.4**

- [x] 4. Checkpoint - Verify preprocessing and split
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Refactor model to use NumPy arrays and tests
  - [x] 5.1 Refactor `src/models/model.py` to match design
    - Remove `build_model` and sklearn Pipeline approach
    - Implement `train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression`
    - Implement `predict(model: LogisticRegression, X: np.ndarray) -> np.ndarray`
    - Model takes NumPy count matrices directly, not text Series
    - _Requirements: 7.1, 7.2, 8.1, 8.2, 8.3_

  - [x]* 5.2 Write property tests for predict
    - **Property 12: Prediction output invariant** — returns array of length len(X) with all values in {0, 1}
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [x] 6. Implement evaluation metrics tests
  - [x] 6.1 Verify `src/evaluation/metrics.py` implementation
    - Implementation is already complete
    - Verify `evaluate_model` handles division-by-zero correctly (returns 0.0)
    - Verify `print_report` outputs formatted table
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

  - [x]* 6.2 Write property tests for evaluation metrics
    - **Property 13: Metrics output validity** — all values in [0.0, 1.0] for any binary arrays of equal length
    - **Validates: Requirements 9.5, 9.7**
    - Use Hypothesis to generate binary arrays

- [x] 7. Wire orchestration and main.py
  - [x] 7.1 Update `run_training` in `src/training/train.py` to use new model interface
    - Call `build_vocabulary` on training texts
    - Call `texts_to_matrix` for train and test sets
    - Pass NumPy arrays to `train_model` and `predict`
    - Return results dict including vocab
    - _Requirements: 12.1, 12.2_

  - [x] 7.2 Update `main.py` to match design flow
    - Load data, preprocess, call `run_training`, print report
    - Ensure error propagation with descriptive messages
    - _Requirements: 12.1, 12.2_

  - [x] 7.3 Update `tests/test_predict.py` to match new model interface
    - Tests currently use Pipeline/Series interface
    - Refactor to test with NumPy arrays as input
    - _Requirements: 8.1, 8.2, 8.3_

- [x] 8. Integration test and package exports
  - [x] 8.1 Write integration test for full pipeline
    - Create a small fixture CSV (5–10 rows) in tests/
    - Test load → preprocess → train → evaluate end-to-end
    - Assert metrics dict returned with expected keys and valid values
    - _Requirements: 12.1_

  - [x] 8.2 Add `__init__.py` exports for all packages
    - `src/data/__init__.py`: export `load_data`, `validate_columns`
    - `src/preprocessing/__init__.py`: export `clean_text`, `normalize_label`, `preprocess_dataset`, `build_vocabulary`, `texts_to_matrix`
    - `src/models/__init__.py`: export `train_model`, `predict`
    - `src/training/__init__.py`: export `split_dataset`, `run_training`
    - `src/evaluation/__init__.py`: export `evaluate_model`, `print_report`
    - `src/utils/__init__.py`: export all config constants
    - _Requirements: 10.1, 10.3_

- [x] 9. Final checkpoint - All tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- `src/utils/config.py` is already complete — no implementation needed
- `src/data/loader.py` and `src/evaluation/metrics.py` are already implemented — tasks verify and test them
- `src/models/model.py` needs a refactor: remove Pipeline, use direct LogisticRegression on NumPy arrays
- Property tests use the Hypothesis library (already present in `.hypothesis/` directory)
- Each task references specific requirements for traceability

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "6.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "2.3", "3.1", "6.2"] },
    { "id": 2, "tasks": ["2.4", "3.2", "5.1"] },
    { "id": 3, "tasks": ["5.2", "7.1"] },
    { "id": 4, "tasks": ["7.2", "7.3"] },
    { "id": 5, "tasks": ["8.1", "8.2"] }
  ]
}
```
