"""
Unit tests for feature_engineering module.

Tests the sklearn-compatible transformers and pipeline components
for Task 3: Feature Engineering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    TemporalFeatureExtractor,
    AggregateFeatureCreator,
    WoETransformer,
    create_feature_engineering_pipeline,
    process_features,
)


@pytest.fixture
def sample_transaction_data():
    """Fixture providing sample transaction data."""
    np.random.seed(42)
    n_samples = 200

    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="h")

    return pd.DataFrame(
        {
            "TransactionId": [f"TXN_{i:04d}" for i in range(n_samples)],
            "CustomerId": np.random.choice(
                ["CUST_A", "CUST_B", "CUST_C", "CUST_D"], n_samples
            ),
            "TransactionStartTime": dates,
            "Value": np.random.exponential(scale=1000, size=n_samples),
            "Amount": np.random.exponential(scale=500, size=n_samples),
            "ProductCategory": np.random.choice(
                ["Electronics", "Clothing", "Food"], n_samples
            ),
            "ChannelId": np.random.choice([1, 2, 3], n_samples),
        }
    )


@pytest.fixture
def sample_target():
    """Fixture providing sample binary target."""
    np.random.seed(42)
    return pd.Series(
        np.random.choice([0, 1], size=200, p=[0.7, 0.3]), name="is_high_risk"
    )


# ============================================================================
# TemporalFeatureExtractor Tests
# ============================================================================


def test_temporal_feature_extractor_initialization():
    """Test TemporalFeatureExtractor initialization."""
    extractor = TemporalFeatureExtractor(datetime_col="TransactionStartTime")
    assert extractor.datetime_col == "TransactionStartTime"


def test_temporal_feature_extractor_transform(sample_transaction_data):
    """Test temporal feature extraction."""
    extractor = TemporalFeatureExtractor(datetime_col="TransactionStartTime")
    extractor.fit(sample_transaction_data)

    result = extractor.transform(sample_transaction_data)

    # Check that temporal features were created
    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns
    assert "transaction_month" in result.columns
    assert "transaction_year" in result.columns
    assert "is_weekend" in result.columns
    assert "time_period" in result.columns

    # Verify value ranges
    assert result["transaction_hour"].between(0, 23).all()
    assert result["transaction_day"].between(0, 6).all()
    assert result["transaction_month"].between(1, 12).all()
    assert result["is_weekend"].isin([0, 1]).all()


def test_temporal_feature_extractor_missing_column():
    """Test behavior when datetime column is missing."""
    extractor = TemporalFeatureExtractor(datetime_col="NonExistentColumn")

    df = pd.DataFrame({"A": [1, 2, 3]})
    result = extractor.fit_transform(df)

    # Should return original dataframe without errors
    assert result.equals(df)


def test_temporal_feature_time_periods(sample_transaction_data):
    """Test time period categorization."""
    extractor = TemporalFeatureExtractor()
    result = extractor.fit_transform(sample_transaction_data)

    # Check time period categories
    expected_periods = ["night", "morning", "afternoon", "evening"]
    assert all(
        period in expected_periods for period in result["time_period"].dropna().unique()
    )


# ============================================================================
# AggregateFeatureCreator Tests
# ============================================================================


def test_aggregate_feature_creator_initialization():
    """Test AggregateFeatureCreator initialization."""
    creator = AggregateFeatureCreator(
        group_col="CustomerId",
        value_col="Value",
        agg_functions=["sum", "mean", "count"],
    )
    assert creator.group_col == "CustomerId"
    assert creator.value_col == "Value"
    assert creator.agg_functions == ["sum", "mean", "count"]


def test_aggregate_feature_creator_fit(sample_transaction_data):
    """Test fitting aggregate feature creator."""
    creator = AggregateFeatureCreator()
    creator.fit(sample_transaction_data)

    # Check that aggregate dataframe was created
    assert creator.agg_df_ is not None
    assert "CustomerId" in creator.agg_df_.columns
    assert "total_transaction_value" in creator.agg_df_.columns
    assert "avg_transaction_value" in creator.agg_df_.columns
    assert "transaction_count" in creator.agg_df_.columns


def test_aggregate_feature_creator_transform(sample_transaction_data):
    """Test transforming data with aggregate features."""
    creator = AggregateFeatureCreator()
    creator.fit(sample_transaction_data)

    result = creator.transform(sample_transaction_data)

    # Check that aggregate columns were merged
    assert "total_transaction_value" in result.columns
    assert "avg_transaction_value" in result.columns
    assert "std_transaction_value" in result.columns
    assert "transaction_count" in result.columns

    # Check that shape increased (new columns added)
    assert result.shape[1] > sample_transaction_data.shape[1]


def test_aggregate_feature_creator_derived_features(sample_transaction_data):
    """Test that derived features are created."""
    creator = AggregateFeatureCreator()
    creator.fit(sample_transaction_data)

    assert "value_range" in creator.agg_df_.columns
    assert "value_cv" in creator.agg_df_.columns  # Coefficient of variation


def test_aggregate_feature_creator_missing_columns():
    """Test behavior with missing required columns."""
    creator = AggregateFeatureCreator(
        group_col="NonExistent", value_col="AlsoNonExistent"
    )

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = creator.fit_transform(df)

    # Should return original dataframe
    assert result.equals(df)


# ============================================================================
# WoETransformer Tests
# ============================================================================


def test_woe_transformer_initialization():
    """Test WoETransformer initialization."""
    transformer = WoETransformer(features=["Feature1", "Feature2"])
    assert transformer.features == ["Feature1", "Feature2"]
    assert transformer.epsilon == 1e-10


def test_woe_transformer_fit(sample_transaction_data, sample_target):
    """Test fitting WoE transformer."""
    transformer = WoETransformer(features=["ProductCategory", "ChannelId"])
    transformer.fit(sample_transaction_data, sample_target)

    # Check that WoE mappings were created
    assert "ProductCategory" in transformer.woe_mappings_
    assert "ChannelId" in transformer.woe_mappings_

    # Check that mappings contain valid WoE values
    for category in sample_transaction_data["ProductCategory"].unique():
        assert category in transformer.woe_mappings_["ProductCategory"]


def test_woe_transformer_transform(sample_transaction_data, sample_target):
    """Test transforming data with WoE encoding."""
    transformer = WoETransformer(features=["ProductCategory"])
    transformer.fit(sample_transaction_data, sample_target)

    result = transformer.transform(sample_transaction_data)

    # Check that WoE column was created
    assert "ProductCategory_woe" in result.columns

    # Check that WoE values are numeric
    assert pd.api.types.is_numeric_dtype(result["ProductCategory_woe"])

    # Check that there are no NaN values (unseen categories mapped to 0)
    assert result["ProductCategory_woe"].notna().all()


def test_woe_transformer_auto_detect_features(sample_transaction_data, sample_target):
    """Test auto-detection of categorical features."""
    transformer = WoETransformer(features=None)  # Auto-detect
    transformer.fit(sample_transaction_data, sample_target)

    # Should have detected categorical features
    assert len(transformer.woe_mappings_) > 0


def test_woe_transformer_unseen_categories(sample_transaction_data, sample_target):
    """Test handling of unseen categories during transform."""
    # Split data
    train_data = sample_transaction_data.iloc[:150].copy()
    test_data = sample_transaction_data.iloc[150:].copy()

    # Add unseen category to test data
    test_data.loc[test_data.index[0], "ProductCategory"] = "UnseenCategory"

    transformer = WoETransformer(features=["ProductCategory"])
    transformer.fit(train_data, sample_target[:150])

    result = transformer.transform(test_data)

    # Unseen category should be mapped to 0
    unseen_woe = result.loc[test_data.index[0], "ProductCategory_woe"]
    assert unseen_woe == 0.0


def test_woe_transformer_single_class_target(sample_transaction_data):
    """Test behavior when target has only one class."""
    single_class_target = pd.Series([1] * len(sample_transaction_data), name="target")

    transformer = WoETransformer(features=["ProductCategory"])
    transformer.fit(sample_transaction_data, single_class_target)

    result = transformer.transform(sample_transaction_data)

    # Should handle gracefully (all WoE = 0)
    assert "ProductCategory_woe" in result.columns
    assert (result["ProductCategory_woe"] == 0.0).all()


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


def test_create_feature_engineering_pipeline():
    """Test creating complete feature engineering pipeline."""
    pipeline = create_feature_engineering_pipeline(
        temporal_col="TransactionStartTime",
        group_col="CustomerId",
        value_col="Value",
        categorical_features=["ProductCategory"],
        use_woe=True,
    )

    # Check that pipeline is a sklearn Pipeline
    assert isinstance(pipeline, Pipeline)

    # Check that pipeline has expected steps
    step_names = [name for name, _ in pipeline.steps]
    assert "temporal_features" in step_names
    assert "aggregate_features" in step_names
    assert "woe_encoding" in step_names


def test_create_pipeline_without_woe():
    """Test creating pipeline without WoE encoding."""
    pipeline = create_feature_engineering_pipeline(use_woe=False)

    step_names = [name for name, _ in pipeline.steps]
    assert "woe_encoding" not in step_names


def test_pipeline_fit_transform(sample_transaction_data, sample_target):
    """Test fitting and transforming with complete pipeline."""
    pipeline = create_feature_engineering_pipeline(
        categorical_features=["ProductCategory"], use_woe=True
    )

    # Fit pipeline
    pipeline.fit(sample_transaction_data, sample_target)

    # Transform data
    result = pipeline.transform(sample_transaction_data)

    # Check that transformations were applied
    assert "transaction_hour" in result.columns  # Temporal features
    assert "total_transaction_value" in result.columns  # Aggregate features
    assert "ProductCategory_woe" in result.columns  # WoE encoding


def test_process_features_function(sample_transaction_data, sample_target):
    """Test the process_features convenience function."""
    result = process_features(
        df=sample_transaction_data,
        target=sample_target,
        categorical_features=["ProductCategory", "ChannelId"],
        use_woe=True,
    )

    # Check that all feature types were created
    assert "transaction_hour" in result.columns
    assert "total_transaction_value" in result.columns
    assert "ProductCategory_woe" in result.columns


def test_pipeline_reproducibility(sample_transaction_data, sample_target):
    """Test that pipeline produces consistent results."""
    pipeline = create_feature_engineering_pipeline(
        categorical_features=["ProductCategory"], use_woe=True
    )

    # Transform twice
    pipeline.fit(sample_transaction_data, sample_target)
    result1 = pipeline.transform(sample_transaction_data)
    result2 = pipeline.transform(sample_transaction_data)

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)


def test_pipeline_with_subset_of_features(sample_transaction_data):
    """Test pipeline with only some transformation steps."""
    # Create pipeline without WoE
    pipeline = Pipeline(
        [
            ("temporal", TemporalFeatureExtractor()),
            ("aggregates", AggregateFeatureCreator()),
        ]
    )

    result = pipeline.fit_transform(sample_transaction_data)

    # Should have temporal and aggregate features
    assert "transaction_hour" in result.columns
    assert "total_transaction_value" in result.columns


def test_pipeline_preserves_original_columns(sample_transaction_data, sample_target):
    """Test that pipeline preserves original columns."""
    original_cols = sample_transaction_data.columns.tolist()

    pipeline = create_feature_engineering_pipeline(
        categorical_features=["ProductCategory"], use_woe=True
    )

    result = pipeline.fit(sample_transaction_data, sample_target).transform(
        sample_transaction_data
    )

    # Original columns should still be present
    for col in original_cols:
        assert col in result.columns


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_empty_dataframe():
    """Test pipeline with empty DataFrame."""
    empty_df = pd.DataFrame()
    pipeline = create_feature_engineering_pipeline(use_woe=False)

    # Should handle gracefully without errors
    result = pipeline.fit_transform(empty_df)
    assert isinstance(result, pd.DataFrame)


def test_single_customer_aggregation():
    """Test aggregation with single customer."""
    df = pd.DataFrame(
        {
            "CustomerId": ["CUST_A"] * 10,
            "Value": np.random.rand(10) * 1000,
            "TransactionStartTime": pd.date_range("2024-01-01", periods=10, freq="h"),
        }
    )

    creator = AggregateFeatureCreator()
    result = creator.fit_transform(df)

    # Should work with single customer
    assert "total_transaction_value" in result.columns
    assert result["CustomerId"].nunique() == 1


def test_all_missing_values():
    """Test WoE transformer with all missing values in feature."""
    df = pd.DataFrame({"feature": [np.nan] * 100, "target": [0, 1] * 50})

    transformer = WoETransformer(features=["feature"])

    # Should handle gracefully
    transformer.fit(df, df["target"])
    result = transformer.transform(df)

    assert "feature_woe" in result.columns
