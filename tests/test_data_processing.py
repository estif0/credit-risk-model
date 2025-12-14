"""
Unit tests for data_processing module.

Tests the DataProcessor class and utility functions for feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing import (
    DataProcessor,
    apply_log_transformation,
    save_processed_data,
)


@pytest.fixture
def sample_transaction_data():
    """Fixture providing sample transaction data for testing."""
    np.random.seed(42)
    n_samples = 100

    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="H")

    return pd.DataFrame(
        {
            "TransactionId": [f"TXN_{i:04d}" for i in range(n_samples)],
            "CustomerId": np.random.choice(
                ["CUST_001", "CUST_002", "CUST_003"], n_samples
            ),
            "TransactionStartTime": dates,
            "Amount": np.random.exponential(scale=1000, size=n_samples),
            "Value": np.random.exponential(scale=1000, size=n_samples),
            "CountryCode": [256] * n_samples,
            "PricingStrategy": np.random.randint(1, 4, n_samples),
        }
    )


def test_data_processor_initialization():
    """Test DataProcessor initialization with different parameters."""
    processor = DataProcessor(scale_features=True, encoding_strategy="label")
    assert processor.scale_features is True
    assert processor.encoding_strategy == "label"
    assert isinstance(processor.scalers, dict)
    assert isinstance(processor.encoders, dict)


def test_load_data_missing_file():
    """Test that load_data raises FileNotFoundError for non-existent file."""
    processor = DataProcessor()
    with pytest.raises(FileNotFoundError):
        processor.load_data("nonexistent_file.csv")


def test_extract_temporal_features(sample_transaction_data):
    """Test temporal feature extraction from timestamp column."""
    processor = DataProcessor()
    result = processor.extract_temporal_features(sample_transaction_data)

    # Check that temporal features were added
    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns
    assert "transaction_month" in result.columns
    assert "transaction_year" in result.columns
    assert "is_weekend" in result.columns

    # Validate ranges
    assert result["transaction_hour"].between(0, 23).all()
    assert result["transaction_day"].between(0, 6).all()
    assert result["transaction_month"].between(1, 12).all()
    assert result["is_weekend"].isin([0, 1]).all()


def test_extract_temporal_features_missing_column(sample_transaction_data):
    """Test that extract_temporal_features raises KeyError for missing column."""
    processor = DataProcessor()
    df_no_time = sample_transaction_data.drop("TransactionStartTime", axis=1)

    with pytest.raises(KeyError):
        processor.extract_temporal_features(df_no_time)


def test_create_aggregate_features(sample_transaction_data):
    """Test aggregate feature creation per customer."""
    processor = DataProcessor()
    result = processor.create_aggregate_features(sample_transaction_data)

    # Check aggregate columns exist
    assert "total_transaction_value" in result.columns
    assert "avg_transaction_value" in result.columns
    assert "std_transaction_value" in result.columns
    assert "transaction_count" in result.columns

    # Verify aggregation worked
    assert len(result) == sample_transaction_data["CustomerId"].nunique()
    assert result["transaction_count"].min() > 0


def test_handle_missing_values():
    """Test missing value handling with different strategies."""
    df = pd.DataFrame(
        {"A": [1, 2, np.nan, 4], "B": [5, np.nan, np.nan, 8], "C": ["x", "y", "z", "w"]}
    )

    processor = DataProcessor()

    # Test mean strategy
    result_mean = processor.handle_missing_values(df, strategy="mean")
    assert result_mean["A"].isnull().sum() == 0

    # Test drop strategy
    result_drop = processor.handle_missing_values(df, strategy="drop")
    assert result_drop.isnull().sum().sum() == 0
    assert len(result_drop) < len(df)


def test_apply_log_transformation():
    """Test log transformation on skewed features."""
    df = pd.DataFrame({"Amount": [100, 1000, 10000], "Value": [50, 500, 5000]})

    result = apply_log_transformation(df, ["Amount", "Value"])

    # Check log columns were created
    assert "Amount_log" in result.columns
    assert "Value_log" in result.columns

    # Verify transformation reduces range
    assert result["Amount_log"].max() < result["Amount"].max()
    assert result["Value_log"].max() < result["Value"].max()


def test_scale_numerical_features(sample_transaction_data):
    """Test numerical feature scaling."""
    processor = DataProcessor(scale_features=True)
    result = processor.scale_numerical_features(
        sample_transaction_data, ["Amount", "Value"]
    )

    # Check scaled columns were created
    assert "Amount_scaled" in result.columns
    assert "Value_scaled" in result.columns


def test_encode_categorical_features(sample_transaction_data):
    """Test categorical feature encoding."""
    processor = DataProcessor(encoding_strategy="label")
    result = processor.encode_categorical_features(
        sample_transaction_data, ["CustomerId"]
    )

    # Check encoded column was created
    assert "CustomerId_encoded" in result.columns
    assert result["CustomerId_encoded"].dtype in ["int64", "int32"]


def test_save_processed_data_creates_file(sample_transaction_data, tmp_path):
    """Test that save_processed_data creates a file."""
    output_file = tmp_path / "test_output.csv"
    save_processed_data(sample_transaction_data, str(output_file))

    assert output_file.exists()

    # Verify data integrity
    loaded = pd.read_csv(output_file)
    assert len(loaded) == len(sample_transaction_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
