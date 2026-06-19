"""Unit tests for the preprocessing module."""

from src.preprocessing.transform import clean_text, normalize_label


def test_clean_text_converts_to_lowercase() -> None:
    """clean_text should convert all characters to lowercase."""
    assert clean_text("GREAT Product") == "great product"


def test_clean_text_removes_extra_spaces() -> None:
    """clean_text should collapse multiple spaces into one."""
    assert clean_text("too   many   spaces") == "too many spaces"


def test_clean_text_removes_punctuation() -> None:
    """clean_text should remove punctuation and special characters."""
    result = clean_text("Wow!!! Great product... #1")
    assert "!" not in result
    assert "." not in result
    assert "#" not in result


def test_normalize_label_positive() -> None:
    """normalize_label should return 1 for ratings >= 4."""
    assert normalize_label(4) == 1
    assert normalize_label(5) == 1


def test_normalize_label_negative() -> None:
    """normalize_label should return 0 for ratings <= 2."""
    assert normalize_label(1) == 0
    assert normalize_label(2) == 0


def test_normalize_label_neutral() -> None:
    """normalize_label should return None for rating == 3."""
    assert normalize_label(3) is None
