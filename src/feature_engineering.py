"""
Feature Engineering Pipeline Module.

This module provides sklearn-compatible transformers and a complete pipeline
for credit risk feature engineering, including temporal features, aggregations,
encoding, scaling, and Weight of Evidence (WoE) transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from datetime column.

    Creates features: hour, day_of_week, month, year, is_weekend, time_period.
    """

    def __init__(self, datetime_col: str = "TransactionStartTime"):
        """
        Initialize temporal feature extractor.

        Args:
            datetime_col: Name of datetime column to extract features from
        """
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        """Fit the transformer (no-op for stateless transformer)."""
        return self

    def transform(self, X):
        """
        Extract temporal features from datetime column.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with added temporal features
        """
        X_copy = X.copy()

        if self.datetime_col not in X_copy.columns:
            logger.warning(
                f"Column {self.datetime_col} not found, skipping temporal extraction"
            )
            return X_copy

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(X_copy[self.datetime_col]):
            X_copy[self.datetime_col] = pd.to_datetime(X_copy[self.datetime_col])

        # Extract temporal components
        X_copy["transaction_hour"] = X_copy[self.datetime_col].dt.hour
        X_copy["transaction_day"] = X_copy[self.datetime_col].dt.dayofweek
        X_copy["transaction_month"] = X_copy[self.datetime_col].dt.month
        X_copy["transaction_year"] = X_copy[self.datetime_col].dt.year

        # Create derived features
        X_copy["is_weekend"] = (X_copy["transaction_day"] >= 5).astype(int)

        # Time period bins (0-6: night, 6-12: morning, 12-18: afternoon, 18-24: evening)
        X_copy["time_period"] = pd.cut(
            X_copy["transaction_hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            include_lowest=True,
        )

        logger.info(f"Extracted temporal features from {self.datetime_col}")
        return X_copy


class AggregateFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create aggregate features per customer.

    Calculates: total, mean, std, min, max, count of transaction values.
    """

    def __init__(
        self,
        group_col: str = "CustomerId",
        value_col: str = "Value",
        agg_functions: Optional[List[str]] = None,
    ):
        """
        Initialize aggregate feature creator.

        Args:
            group_col: Column to group by (e.g., CustomerId)
            value_col: Column to aggregate (e.g., Value)
            agg_functions: List of aggregation functions to apply
        """
        self.group_col = group_col
        self.value_col = value_col
        self.agg_functions = agg_functions or [
            "sum",
            "mean",
            "std",
            "min",
            "max",
            "count",
        ]
        self.agg_df_ = None

    def fit(self, X, y=None):
        """
        Fit by computing aggregate features.

        Args:
            X: Input DataFrame
            y: Target variable (unused)

        Returns:
            self
        """
        if self.group_col not in X.columns or self.value_col not in X.columns:
            logger.warning(f"Required columns not found, skipping aggregation")
            return self

        # Calculate aggregates
        self.agg_df_ = (
            X.groupby(self.group_col)[self.value_col]
            .agg(self.agg_functions)
            .reset_index()
        )

        # Rename columns
        col_mapping = {
            "sum": "total_transaction_value",
            "mean": "avg_transaction_value",
            "std": "std_transaction_value",
            "min": "min_transaction_value",
            "max": "max_transaction_value",
            "count": "transaction_count",
        }

        for old_name, new_name in col_mapping.items():
            if old_name in self.agg_df_.columns:
                self.agg_df_.rename(columns={old_name: new_name}, inplace=True)

        # Handle NaN in std (occurs when only 1 transaction)
        if "std_transaction_value" in self.agg_df_.columns:
            self.agg_df_["std_transaction_value"].fillna(0, inplace=True)

        # Create derived features
        if (
            "max_transaction_value" in self.agg_df_.columns
            and "min_transaction_value" in self.agg_df_.columns
        ):
            self.agg_df_["value_range"] = (
                self.agg_df_["max_transaction_value"]
                - self.agg_df_["min_transaction_value"]
            )

        # Coefficient of variation
        if (
            "std_transaction_value" in self.agg_df_.columns
            and "avg_transaction_value" in self.agg_df_.columns
        ):
            self.agg_df_["value_cv"] = np.where(
                self.agg_df_["avg_transaction_value"] > 0,
                self.agg_df_["std_transaction_value"]
                / self.agg_df_["avg_transaction_value"],
                0,
            )

        logger.info(f"Computed aggregate features for {len(self.agg_df_)} groups")
        return self

    def transform(self, X):
        """
        Merge aggregate features into input DataFrame.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with merged aggregate features
        """
        if self.agg_df_ is None:
            logger.warning("Transformer not fitted, returning original DataFrame")
            return X

        X_copy = X.copy()

        # Merge aggregate features
        X_merged = X_copy.merge(self.agg_df_, on=self.group_col, how="left")

        logger.info(f"Merged aggregate features into DataFrame")
        return X_merged


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) Transformer for categorical features.

    Transforms categorical features using WoE encoding, which is particularly
    useful for credit risk modeling as it captures the relationship between
    feature values and the target variable.
    """

    def __init__(self, features: Optional[List[str]] = None, epsilon: float = 1e-10):
        """
        Initialize WoE transformer.

        Args:
            features: List of categorical features to transform
            epsilon: Small constant to avoid division by zero
        """
        self.features = features
        self.epsilon = epsilon
        self.woe_mappings_: Dict[str, Dict] = {}

    def fit(self, X, y):
        """
        Fit WoE encoder by calculating WoE values for each feature.

        Args:
            X: Input DataFrame with categorical features
            y: Binary target variable (0/1)

        Returns:
            self
        """
        if self.features is None:
            # Auto-detect categorical features
            self.features = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        for feature in self.features:
            if feature not in X.columns:
                logger.warning(f"Feature {feature} not found, skipping")
                continue

            # Calculate WoE for each category
            woe_map = self._calculate_woe(X[feature], y)
            self.woe_mappings_[feature] = woe_map

        logger.info(f"Fitted WoE transformer for {len(self.woe_mappings_)} features")
        return self

    def _calculate_woe(self, feature_series: pd.Series, target: pd.Series) -> Dict:
        """
        Calculate WoE for a single feature.

        Args:
            feature_series: Categorical feature values
            target: Binary target variable

        Returns:
            Dictionary mapping feature values to WoE scores
        """
        # Create crosstab
        df_temp = pd.DataFrame({"feature": feature_series, "target": target})

        # Group by feature and calculate event/non-event counts
        grouped = df_temp.groupby("feature")["target"].agg(["sum", "count"])
        grouped["non_event"] = grouped["count"] - grouped["sum"]
        grouped.rename(columns={"sum": "event"}, inplace=True)

        # Calculate distribution of events and non-events
        total_events = grouped["event"].sum()
        total_non_events = grouped["non_event"].sum()

        # Avoid division by zero
        if total_events == 0 or total_non_events == 0:
            logger.warning("Target has only one class, returning zero WoE")
            return {cat: 0.0 for cat in feature_series.unique()}

        grouped["dist_event"] = (grouped["event"] + self.epsilon) / total_events
        grouped["dist_non_event"] = (
            grouped["non_event"] + self.epsilon
        ) / total_non_events

        # Calculate WoE
        grouped["woe"] = np.log(grouped["dist_event"] / grouped["dist_non_event"])

        # Convert to dictionary
        woe_map = grouped["woe"].to_dict()

        # Calculate IV for logging
        grouped["iv"] = (grouped["dist_event"] - grouped["dist_non_event"]) * grouped[
            "woe"
        ]
        total_iv = grouped["iv"].sum()

        logger.debug(f"Feature IV: {total_iv:.4f}")

        return woe_map

    def transform(self, X):
        """
        Transform features using fitted WoE mappings.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with WoE-encoded features
        """
        X_copy = X.copy()

        for feature, woe_map in self.woe_mappings_.items():
            if feature not in X_copy.columns:
                logger.warning(f"Feature {feature} not found in transform data")
                continue

            # Create WoE column
            woe_col_name = f"{feature}_woe"
            X_copy[woe_col_name] = X_copy[feature].map(woe_map)

            # Handle unseen categories (map to 0)
            X_copy[woe_col_name].fillna(0, inplace=True)

        logger.info(
            f"Transformed {len(self.woe_mappings_)} features using WoE encoding"
        )
        return X_copy


def create_feature_engineering_pipeline(
    temporal_col: str = "TransactionStartTime",
    group_col: str = "CustomerId",
    value_col: str = "Value",
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    use_woe: bool = True,
) -> Pipeline:
    """
    Create a complete sklearn Pipeline for feature engineering.

    This pipeline chains all transformation steps:
    1. Temporal feature extraction
    2. Aggregate feature creation
    3. WoE encoding (optional)
    4. Numerical feature scaling

    Args:
        temporal_col: Datetime column name
        group_col: Column to group by for aggregation
        value_col: Column to aggregate
        categorical_features: List of categorical features for WoE
        numerical_features: List of numerical features to scale
        use_woe: Whether to include WoE transformation

    Returns:
        sklearn Pipeline with all transformers
    """
    steps = [
        ("temporal_features", TemporalFeatureExtractor(datetime_col=temporal_col)),
        (
            "aggregate_features",
            AggregateFeatureCreator(group_col=group_col, value_col=value_col),
        ),
    ]

    # Add WoE transformer if specified
    if use_woe and categorical_features:
        steps.append(("woe_encoding", WoETransformer(features=categorical_features)))

    pipeline = Pipeline(steps)

    logger.info(f"Created feature engineering pipeline with {len(steps)} steps")
    return pipeline


def process_features(
    df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    temporal_col: str = "TransactionStartTime",
    group_col: str = "CustomerId",
    value_col: str = "Value",
    categorical_features: Optional[List[str]] = None,
    use_woe: bool = True,
) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline to data.

    Args:
        df: Input DataFrame
        target: Target variable (required for WoE)
        temporal_col: Datetime column name
        group_col: Column to group by
        value_col: Column to aggregate
        categorical_features: Features for WoE encoding
        use_woe: Whether to use WoE encoding

    Returns:
        Transformed DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline")

    # Create pipeline
    pipeline = create_feature_engineering_pipeline(
        temporal_col=temporal_col,
        group_col=group_col,
        value_col=value_col,
        categorical_features=categorical_features if use_woe else None,
        use_woe=use_woe,
    )

    # Fit and transform
    if use_woe and target is not None:
        # WoE requires target variable
        pipeline.fit(df, target)
    else:
        pipeline.fit(df)

    df_transformed = pipeline.transform(df)

    logger.info(f"Feature engineering complete. Shape: {df_transformed.shape}")
    return df_transformed


if __name__ == "__main__":
    # Example usage
    from src.data_processing import DataProcessor

    try:
        # Load data
        processor = DataProcessor()
        df = processor.load_data("data/raw/data.csv")

        logger.info(f"Loaded {len(df)} transactions")

        # Define categorical features for WoE
        categorical_features = ["ProductCategory", "ChannelId"]

        # Create dummy target for demonstration
        # In practice, this would come from RFM analysis
        target = pd.Series(np.random.randint(0, 2, len(df)), name="is_high_risk")

        # Apply feature engineering
        df_transformed = process_features(
            df=df,
            target=target,
            categorical_features=categorical_features,
            use_woe=True,
        )

        logger.info(f"Features created: {df_transformed.shape[1]} columns")
        logger.info(f"Sample features: {df_transformed.columns.tolist()[:20]}")

        # Save processed data
        processor.save_processed_data(
            df_transformed, "data/processed/features_with_pipeline.csv"
        )

        logger.info("Feature engineering pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
