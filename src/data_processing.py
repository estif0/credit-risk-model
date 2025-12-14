"""
Data Processing Module for Credit Risk Model.

This module provides classes and functions for feature engineering,
including temporal feature extraction, aggregate features, and encoding strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Main data processing class for credit risk feature engineering.

    This class handles loading, cleaning, and transforming raw transaction data
    into features suitable for modeling, following sklearn's transformer API.

    Attributes:
        scalers (Dict): Dictionary of fitted scalers for numerical features
        encoders (Dict): Dictionary of fitted encoders for categorical features
        feature_names (List[str]): List of generated feature names
    """

    def __init__(self, scale_features: bool = True, encoding_strategy: str = "label"):
        """
        Initialize the DataProcessor.

        Args:
            scale_features: Whether to apply scaling to numerical features
            encoding_strategy: Strategy for categorical encoding ('label', 'onehot', or 'target')
        """
        self.scale_features = scale_features
        self.encoding_strategy = encoding_strategy
        self.scalers: Dict = {}
        self.encoders: Dict = {}
        self.feature_names: List[str] = []

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load transaction data from CSV with error handling.

        Args:
            filepath: Path to the CSV file

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or has incorrect format
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)

            if df.empty:
                raise ValueError("Loaded DataFrame is empty")

            logger.info(
                f"Successfully loaded {len(df)} transactions with {len(df.columns)} features"
            )
            return df

        except FileNotFoundError as e:
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Data file not found at {filepath}") from e

        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data file: {filepath}")
            raise ValueError(f"Data file is empty: {filepath}") from e

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def extract_temporal_features(
        self, df: pd.DataFrame, time_column: str = "TransactionStartTime"
    ) -> pd.DataFrame:
        """
        Extract temporal features from timestamp column.

        Args:
            df: Input DataFrame
            time_column: Name of the timestamp column

        Returns:
            DataFrame with added temporal features

        Raises:
            KeyError: If time_column doesn't exist in DataFrame
            ValueError: If time_column cannot be converted to datetime
        """
        try:
            if time_column not in df.columns:
                raise KeyError(f"Column '{time_column}' not found in DataFrame")

            logger.info(f"Extracting temporal features from {time_column}")
            df_copy = df.copy()

            # Convert to datetime
            try:
                df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            except Exception as e:
                raise ValueError(
                    f"Cannot convert {time_column} to datetime: {str(e)}"
                ) from e

            # Extract temporal components
            df_copy["transaction_hour"] = df_copy[time_column].dt.hour
            df_copy["transaction_day"] = df_copy[time_column].dt.dayofweek
            df_copy["transaction_month"] = df_copy[time_column].dt.month
            df_copy["transaction_year"] = df_copy[time_column].dt.year
            df_copy["transaction_date"] = df_copy[time_column].dt.date

            # Create time period bins
            df_copy["time_period"] = pd.cut(
                df_copy["transaction_hour"],
                bins=[0, 6, 12, 18, 24],
                labels=["night", "morning", "afternoon", "evening"],
                include_lowest=True,
            )

            # Weekend flag
            df_copy["is_weekend"] = (df_copy["transaction_day"] >= 5).astype(int)

            logger.info("Successfully extracted temporal features")
            return df_copy

        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
            raise

    def create_aggregate_features(
        self, df: pd.DataFrame, group_by: str = "CustomerId"
    ) -> pd.DataFrame:
        """
        Create aggregate features per customer.

        Args:
            df: Input DataFrame with transaction-level data
            group_by: Column to group by (typically CustomerId or AccountId)

        Returns:
            DataFrame with aggregate features per group

        Raises:
            KeyError: If required columns are missing
        """
        try:
            if group_by not in df.columns:
                raise KeyError(f"Column '{group_by}' not found in DataFrame")

            logger.info(f"Creating aggregate features grouped by {group_by}")

            # Ensure Value column exists
            if "Value" not in df.columns:
                raise KeyError("'Value' column required for aggregation")

            # Calculate aggregates
            agg_features = (
                df.groupby(group_by)
                .agg(
                    {
                        "Value": ["sum", "mean", "std", "min", "max", "count"],
                        "TransactionId": "count",
                    }
                )
                .reset_index()
            )

            # Flatten column names
            agg_features.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0]
                for col in agg_features.columns
            ]

            # Rename for clarity
            agg_features.rename(
                columns={
                    "Value_sum": "total_transaction_value",
                    "Value_mean": "avg_transaction_value",
                    "Value_std": "std_transaction_value",
                    "Value_min": "min_transaction_value",
                    "Value_max": "max_transaction_value",
                    "Value_count": "transaction_count",
                    "TransactionId_count": "transaction_frequency",
                },
                inplace=True,
            )

            # Handle NaN in std (occurs when only 1 transaction)
            agg_features["std_transaction_value"].fillna(0, inplace=True)

            # Create additional derived features
            agg_features["value_range"] = (
                agg_features["max_transaction_value"]
                - agg_features["min_transaction_value"]
            )

            # Coefficient of variation (relative volatility)
            agg_features["value_cv"] = np.where(
                agg_features["avg_transaction_value"] > 0,
                agg_features["std_transaction_value"]
                / agg_features["avg_transaction_value"],
                0,
            )

            logger.info(f"Created aggregate features for {len(agg_features)} groups")
            return agg_features

        except Exception as e:
            logger.error(f"Error creating aggregate features: {str(e)}")
            raise

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "mean"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')

        Returns:
            DataFrame with missing values handled
        """
        try:
            logger.info("Checking for missing values")
            df_copy = df.copy()

            missing_counts = df_copy.isnull().sum()
            missing_features = missing_counts[missing_counts > 0]

            if len(missing_features) == 0:
                logger.info("No missing values found")
                return df_copy

            logger.warning(f"Found missing values in {len(missing_features)} features")

            if strategy == "drop":
                df_copy.dropna(inplace=True)
                logger.info(
                    f"Dropped rows with missing values. Remaining: {len(df_copy)}"
                )

            elif strategy == "mean":
                for col in missing_features.index:
                    if df_copy[col].dtype in ["float64", "int64"]:
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)

            elif strategy == "median":
                for col in missing_features.index:
                    if df_copy[col].dtype in ["float64", "int64"]:
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)

            elif strategy == "mode":
                for col in missing_features.index:
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)

            logger.info(f"Missing values handled using strategy: {strategy}")
            return df_copy

        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise

    def encode_categorical_features(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> pd.DataFrame:
        """
        Encode categorical features using specified strategy.

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names to encode

        Returns:
            DataFrame with encoded categorical features
        """
        try:
            logger.info(f"Encoding {len(categorical_cols)} categorical features")
            df_copy = df.copy()

            for col in categorical_cols:
                if col not in df_copy.columns:
                    logger.warning(f"Column '{col}' not found, skipping")
                    continue

                if self.encoding_strategy == "label":
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df_copy[f"{col}_encoded"] = self.encoders[col].fit_transform(
                            df_copy[col].astype(str)
                        )
                    else:
                        df_copy[f"{col}_encoded"] = self.encoders[col].transform(
                            df_copy[col].astype(str)
                        )

            logger.info("Categorical encoding complete")
            return df_copy

        except Exception as e:
            logger.error(f"Error encoding categorical features: {str(e)}")
            raise

    def scale_numerical_features(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names to scale

        Returns:
            DataFrame with scaled numerical features
        """
        try:
            if not self.scale_features:
                logger.info("Scaling disabled, skipping")
                return df

            logger.info(f"Scaling {len(numerical_cols)} numerical features")
            df_copy = df.copy()

            for col in numerical_cols:
                if col not in df_copy.columns:
                    logger.warning(f"Column '{col}' not found, skipping")
                    continue

                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_copy[f"{col}_scaled"] = self.scalers[col].fit_transform(
                        df_copy[[col]]
                    )
                else:
                    df_copy[f"{col}_scaled"] = self.scalers[col].transform(
                        df_copy[[col]]
                    )

            logger.info("Numerical scaling complete")
            return df_copy

        except Exception as e:
            logger.error(f"Error scaling numerical features: {str(e)}")
            raise

    def fit(self, X, y=None):
        """Fit the processor (sklearn compatibility)."""
        return self

    def transform(self, X):
        """Transform the data (sklearn compatibility)."""
        return X


def apply_log_transformation(
    df: pd.DataFrame, columns: List[str], offset: float = 1.0
) -> pd.DataFrame:
    """
    Apply log transformation to skewed features.

    Args:
        df: Input DataFrame
        columns: List of columns to transform
        offset: Small constant to add before log (handles zeros)

    Returns:
        DataFrame with log-transformed features
    """
    try:
        logger.info(f"Applying log transformation to {len(columns)} features")
        df_copy = df.copy()

        for col in columns:
            if col not in df_copy.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            # Handle negative values by taking absolute value
            df_copy[f"{col}_log"] = np.log1p(np.abs(df_copy[col]) + offset)

        logger.info("Log transformation complete")
        return df_copy

    except Exception as e:
        logger.error(f"Error in log transformation: {str(e)}")
        raise


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save processed data to CSV with error handling.

    Args:
        filepath: Output file path
        df: DataFrame to save

    Raises:
        IOError: If file cannot be written
    """
    try:
        logger.info(f"Saving processed data to {filepath}")
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved {len(df)} rows to {filepath}")

    except IOError as e:
        logger.error(f"Error saving file: {str(e)}")
        raise IOError(f"Cannot write to {filepath}") from e

    except Exception as e:
        logger.error(f"Unexpected error saving data: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    try:
        processor = DataProcessor(scale_features=True)

        # Load data
        df = processor.load_data("../data/raw/data.csv")

        # Extract temporal features
        df = processor.extract_temporal_features(df)

        # Create aggregate features
        agg_df = processor.create_aggregate_features(df)

        # Apply log transformation to skewed features
        df = apply_log_transformation(df, ["Amount", "Value"])

        # Save processed data
        save_processed_data(df, "../data/processed/processed_transactions.csv")
        save_processed_data(agg_df, "../data/processed/aggregate_features.csv")

        logger.info("Data processing pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
