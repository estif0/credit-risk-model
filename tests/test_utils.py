"""
Unit tests for utils module.

Tests utility functions for validation, calculations, and I/O operations.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.utils import (
    validate_dataframe,
    check_null_values,
    detect_outliers_iqr,
    calculate_skewness,
    ensure_directory_exists,
    safe_divide,
    load_config,
    save_config,
    format_large_number,
    calculate_class_weights,
    merge_transaction_features,
)


def test_validate_dataframe_success():
    """Test DataFrame validation with valid input."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    validate_dataframe(df, required_columns=["A", "B"])
    # Should not raise any exception


def test_validate_dataframe_missing_column():
    """Test DataFrame validation raises KeyError for missing columns."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(KeyError):
        validate_dataframe(df, required_columns=["A", "B"])


def test_validate_dataframe_empty():
    """Test DataFrame validation raises ValueError for empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["A"])


def test_check_null_values():
    """Test null value checking returns correct proportions."""
    df = pd.DataFrame(
        {"A": [1, 2, np.nan, 4], "B": [5, np.nan, np.nan, 8], "C": [1, 2, 3, 4]}
    )

    null_props = check_null_values(df, threshold=0.3)

    assert null_props["A"] == 0.25
    assert null_props["B"] == 0.5
    assert null_props["C"] == 0.0


def test_detect_outliers_iqr():
    """Test IQR outlier detection."""
    # Create data with known outliers
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5, 100])  # 100 is outlier
    outliers = detect_outliers_iqr(data)

    assert outliers.sum() > 0  # At least one outlier detected
    assert outliers.iloc[-1] == True  # 100 should be flagged


def test_calculate_skewness():
    """Test skewness calculation for numerical columns."""
    df = pd.DataFrame(
        {
            "normal": np.random.normal(0, 1, 1000),
            "skewed": np.random.exponential(2, 1000),
            "uniform": np.random.uniform(0, 1, 1000),
        }
    )

    skewness = calculate_skewness(df)

    assert "normal" in skewness
    assert "skewed" in skewness
    assert abs(skewness["skewed"]) > abs(skewness["normal"])


def test_ensure_directory_exists(tmp_path):
    """Test directory creation."""
    test_file = tmp_path / "subdir" / "file.txt"
    ensure_directory_exists(test_file)

    assert test_file.parent.exists()


def test_safe_divide():
    """Test safe division handles zero denominators."""
    numerator = np.array([10, 20, 30])
    denominator = np.array([2, 0, 5])

    result = safe_divide(numerator, denominator, default=0.0)

    assert result[0] == 5.0
    assert result[1] == 0.0  # Division by zero handled
    assert result[2] == 6.0


def test_load_config(tmp_path):
    """Test configuration loading from JSON."""
    config_file = tmp_path / "config.json"
    config_data = {"param1": 10, "param2": "value"}

    with open(config_file, "w") as f:
        json.dump(config_data, f)

    loaded_config = load_config(str(config_file))

    assert loaded_config == config_data


def test_load_config_missing_file():
    """Test load_config raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.json")


def test_save_config(tmp_path):
    """Test configuration saving to JSON."""
    config_file = tmp_path / "test_config.json"
    config_data = {"n_clusters": 3, "random_state": 42}

    save_config(config_data, str(config_file))

    assert config_file.exists()

    # Verify content
    with open(config_file, "r") as f:
        loaded = json.load(f)

    assert loaded == config_data


def test_format_large_number():
    """Test large number formatting with suffixes."""
    assert format_large_number(1500) == "1.50K"
    assert format_large_number(1500000) == "1.50M"
    assert format_large_number(1500000000) == "1.50B"
    assert format_large_number(500) == "500.00"


def test_calculate_class_weights():
    """Test class weight calculation for imbalanced data."""
    # Imbalanced dataset: 80% class 0, 20% class 1
    y = pd.Series([0] * 80 + [1] * 20)

    weights = calculate_class_weights(y)

    assert 0 in weights
    assert 1 in weights
    # Minority class should have higher weight
    assert weights[1] > weights[0]


def test_merge_transaction_features():
    """Test merging of transaction, aggregate, and RFM features."""
    # Create sample datasets
    transaction_df = pd.DataFrame(
        {
            "TransactionId": ["T1", "T2", "T3", "T4"],
            "CustomerId": ["C1", "C1", "C2", "C2"],
            "Amount": [100, 200, 300, 400],
        }
    )

    aggregate_df = pd.DataFrame(
        {"CustomerId": ["C1", "C2"], "total_value": [300, 700], "avg_value": [150, 350]}
    )

    rfm_df = pd.DataFrame(
        {
            "CustomerId": ["C1", "C2"],
            "Recency": [5, 10],
            "Frequency": [2, 2],
            "Monetary": [300, 700],
            "is_high_risk": [0, 1],
        }
    )

    merged = merge_transaction_features(transaction_df, aggregate_df, rfm_df)

    # Check all features are present
    assert "total_value" in merged.columns
    assert "Recency" in merged.columns
    assert "is_high_risk" in merged.columns

    # Check correct number of rows (transaction-level)
    assert len(merged) == len(transaction_df)


def test_safe_divide_scalar():
    """Test safe_divide with scalar inputs."""
    result = safe_divide(10, 2)
    assert result == 5.0

    result_zero = safe_divide(10, 0, default=-1)
    assert result_zero == -1


def test_detect_outliers_iqr_no_outliers():
    """Test IQR outlier detection with no outliers."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    outliers = detect_outliers_iqr(data)

    # With uniform data, likely no outliers
    assert isinstance(outliers, pd.Series)
    assert outliers.dtype == bool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
