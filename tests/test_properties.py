"""Property-based tests using Hypothesis for correctness properties."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.metrics import evaluate_model
from src.preprocessing.transform import clean_text, normalize_label, preprocess_dataset
from src.training.train import split_dataset


# ============================================================
# Property 1: Text cleaning idempotency
# clean_text(clean_text(text)) == clean_text(text)
# Validates: Requirements 2.5
# ============================================================
@given(text=st.text(min_size=0, max_size=200))
@settings(max_examples=200)
def test_property_1_clean_text_idempotency(text: str) -> None:
    """Property 1: Applying clean_text twice equals applying it once."""
    once = clean_text(text)
    twice = clean_text(once)
    assert once == twice, f"Idempotency violated: clean_text('{once}') != '{twice}'"


# ============================================================
# Property 2: Clean text output format
# Result contains only lowercase alphanumeric and single spaces
# Validates: Requirements 2.1, 2.2, 2.3
# ============================================================
@given(text=st.text(min_size=0, max_size=200))
@settings(max_examples=200)
def test_property_2_clean_text_output_format(text: str) -> None:
    """Property 2: Output has only lowercase alphanum + single spaces, no leading/trailing whitespace."""
    result = clean_text(text)

    # All lowercase
    assert result == result.lower(), f"Not lowercase: {result}"

    # No leading/trailing whitespace
    assert result == result.strip(), f"Has leading/trailing whitespace: '{result}'"

    # No consecutive spaces
    assert "  " not in result, f"Consecutive spaces found: '{result}'"

    # Only alphanumeric and space characters
    for ch in result:
        assert ch.isalnum() or ch == " ", f"Invalid character '{ch}' in: '{result}'"


# ============================================================
# Property 3: Label determinism and completeness
# normalize_label always returns the same value, in {0, 1, None}
# Validates: Requirements 3.1, 3.2, 3.3, 3.4
# ============================================================
@given(rating=st.integers(min_value=1, max_value=5))
@settings(max_examples=50)
def test_property_3_label_determinism(rating: int) -> None:
    """Property 3: normalize_label is deterministic and complete for ratings 1-5."""
    result1 = normalize_label(rating)
    result2 = normalize_label(rating)

    # Deterministic
    assert result1 == result2, f"Non-deterministic for rating {rating}"

    # Value in {0, 1, None}
    assert result1 in (0, 1, None), f"Invalid result {result1} for rating {rating}"

    # Correct mapping
    if rating >= 4:
        assert result1 == 1
    elif rating <= 2:
        assert result1 == 0
    else:
        assert result1 is None


# ============================================================
# Property 4: Preprocessing row reduction
# Output has no more rows than input
# Validates: Requirements 4.6
# ============================================================
@given(
    n_rows=st.integers(min_value=1, max_value=20),
    data=st.data(),
)
@settings(max_examples=50)
def test_property_4_preprocessing_row_reduction(n_rows: int, data) -> None:
    """Property 4: preprocess_dataset never adds rows."""
    texts = data.draw(
        st.lists(
            st.text(alphabet=st.characters(whitelist_categories=("L", "N", "Zs")), min_size=0, max_size=50),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    ratings = data.draw(
        st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    df = pd.DataFrame({"reviews.text": texts, "reviews.rating": ratings})
    result = preprocess_dataset(df)
    assert len(result) <= len(df), "preprocess_dataset added rows"


# ============================================================
# Property 5: Preprocessing output validity
# All text satisfies clean_text postconditions, sentiment in {0, 1}, no NaN
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.7
# ============================================================
@given(
    n_rows=st.integers(min_value=1, max_value=15),
    data=st.data(),
)
@settings(max_examples=30)
def test_property_5_preprocessing_output_validity(n_rows: int, data) -> None:
    """Property 5: Output has valid text, sentiment in {0,1}, no NaN in key columns."""
    texts = data.draw(
        st.lists(
            st.text(alphabet=st.characters(whitelist_categories=("L", "N", "Zs")), min_size=1, max_size=50),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    ratings = data.draw(
        st.lists(
            st.sampled_from([1, 2, 4, 5]),  # Exclude 3 to guarantee non-empty output
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    df = pd.DataFrame({"reviews.text": texts, "reviews.rating": ratings})
    result = preprocess_dataset(df)

    if len(result) == 0:
        return  # All rows were empty text, valid

    # No NaN in text or sentiment
    assert result["reviews.text"].notna().all()
    assert result["sentiment"].notna().all()

    # Sentiment values in {0, 1}
    assert set(result["sentiment"].unique()).issubset({0, 1})

    # Text satisfies clean_text postconditions
    for text_val in result["reviews.text"]:
        assert text_val == text_val.lower()
        assert text_val == text_val.strip()
        assert "  " not in text_val
        for ch in text_val:
            assert ch.isalnum() or ch == " "


# ============================================================
# Property 6: Split conservation
# len(x_train) + len(x_test) == len(data)
# Validates: Requirements 5.4
# ============================================================
@given(
    n_rows=st.integers(min_value=5, max_value=100),
    test_size=st.floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=50)
def test_property_6_split_conservation(n_rows: int, test_size: float) -> None:
    """Property 6: No rows created or lost during split."""
    df = pd.DataFrame({
        "reviews.text": [f"text {i}" for i in range(n_rows)],
        "sentiment": [i % 2 for i in range(n_rows)],
    })
    x_train, x_test, y_train, y_test = split_dataset(
        df, "reviews.text", "sentiment", test_size
    )
    assert len(x_train) + len(x_test) == n_rows
    assert len(y_train) + len(y_test) == n_rows


# ============================================================
# Property 7: Split disjointness
# Train and test share no data
# Validates: Requirements 5.5
# ============================================================
@given(n_rows=st.integers(min_value=5, max_value=50))
@settings(max_examples=30)
def test_property_7_split_disjointness(n_rows: int) -> None:
    """Property 7: Train and test sets are completely disjoint."""
    df = pd.DataFrame({
        "reviews.text": [f"unique_text_{i}" for i in range(n_rows)],
        "sentiment": [i % 2 for i in range(n_rows)],
    })
    x_train, x_test, _, _ = split_dataset(df, "reviews.text", "sentiment", 0.2)

    train_set = set(x_train.values)
    test_set = set(x_test.values)
    assert train_set.isdisjoint(test_set), "Train and test sets overlap"


# ============================================================
# Property 8: Split size correctness
# Test set size equals int(n * test_size) within ±1
# Validates: Requirements 5.3
# ============================================================
@given(
    n_rows=st.integers(min_value=5, max_value=100),
    test_size=st.floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=50)
def test_property_8_split_size_correctness(n_rows: int, test_size: float) -> None:
    """Property 8: Test set size is int(n * test_size)."""
    df = pd.DataFrame({
        "reviews.text": [f"text {i}" for i in range(n_rows)],
        "sentiment": [i % 2 for i in range(n_rows)],
    })
    _, x_test, _, _ = split_dataset(df, "reviews.text", "sentiment", test_size)
    expected_test_size = int(n_rows * test_size)
    assert abs(len(x_test) - expected_test_size) <= 1


# ============================================================
# Property 9: Split reproducibility
# Same seed produces identical splits
# Validates: Requirements 5.6
# ============================================================
@given(n_rows=st.integers(min_value=5, max_value=50))
@settings(max_examples=30)
def test_property_9_split_reproducibility(n_rows: int) -> None:
    """Property 9: Calling split_dataset twice gives identical results."""
    df = pd.DataFrame({
        "reviews.text": [f"text {i}" for i in range(n_rows)],
        "sentiment": [i % 2 for i in range(n_rows)],
    })
    x_train1, x_test1, y_train1, y_test1 = split_dataset(
        df, "reviews.text", "sentiment", 0.2
    )
    x_train2, x_test2, y_train2, y_test2 = split_dataset(
        df, "reviews.text", "sentiment", 0.2
    )
    pd.testing.assert_series_equal(x_train1, x_train2)
    pd.testing.assert_series_equal(x_test1, x_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


# ============================================================
# Property 10: Prediction output invariant
# predict returns np.ndarray of shape (n,) with values in {0, 1}
# Validates: Requirements 8.1, 8.2, 8.3
# ============================================================
@given(n_samples=st.integers(min_value=1, max_value=20))
@settings(max_examples=20)
def test_property_10_prediction_output_invariant(n_samples: int) -> None:
    """Property 10: predict returns ndarray of shape (n,) with values in {0, 1}."""
    from src.models.model import predict, train_model

    # Train a tiny model
    x_train = pd.Series([
        "great product love it amazing",
        "wonderful quality excellent",
        "terrible awful worst garbage",
        "horrible bad broken",
    ])
    y_train = pd.Series([1, 1, 0, 0])
    model = train_model(x_train, y_train)

    # Generate test inputs
    x_test = pd.Series([f"test text sample {i}" for i in range(n_samples)])
    result = predict(model, x_test)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_samples,)
    assert all(val in (0, 1) for val in result)
    assert np.issubdtype(result.dtype, np.integer)


# ============================================================
# Property 11: Metrics formula correctness
# All metrics in [0.0, 1.0], division-by-zero returns 0.0
# Validates: Requirements 9.4, 9.5, 9.6, 9.7, 9.8
# ============================================================
@given(
    y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=100),
    y_pred=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=100),
)
@settings(max_examples=100)
def test_property_11_metrics_formula_correctness(
    y_true: list[int], y_pred: list[int]
) -> None:
    """Property 11: All metrics in [0, 1], correct formulas, zero-division handled."""
    # Make same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    metrics = evaluate_model(y_true, y_pred)

    # All keys present
    assert set(metrics.keys()) == {"accuracy", "f1_score", "precision", "recall"}

    # All values in [0.0, 1.0]
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"

    # Verify accuracy formula
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    tn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))
    total = tp + tn + fp + fn

    expected_accuracy = (tp + tn) / total if total > 0 else 0.0
    assert abs(metrics["accuracy"] - expected_accuracy) < 1e-10

    expected_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    assert abs(metrics["precision"] - expected_precision) < 1e-10

    expected_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    assert abs(metrics["recall"] - expected_recall) < 1e-10
