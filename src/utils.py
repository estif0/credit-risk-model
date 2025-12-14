"""
Utility Functions for Credit Risk Model.

This module provides helper functions for common tasks including
data validation, metric calculations, file I/O, and logging utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
import os
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that a DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValueError: If DataFrame is empty
        KeyError: If required columns are missing
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    logger.info(f"DataFrame validation passed. Shape: {df.shape}")


def check_null_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """
    Check for null values in DataFrame and report columns exceeding threshold.

    Args:
        df: DataFrame to check
        threshold: Maximum acceptable proportion of nulls (0-1)

    Returns:
        Dictionary mapping column names to null proportions
    """
    null_proportions = df.isnull().mean()
    problematic_cols = null_proportions[null_proportions > threshold]

    if len(problematic_cols) > 0:
        logger.warning(f"Columns with >{threshold*100}% nulls:\n{problematic_cols}")
    else:
        logger.info(f"No columns exceed {threshold*100}% null threshold")

    return null_proportions.to_dict()


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using the IQR method.

    Args:
        series: Pandas Series to check for outliers
        multiplier: IQR multiplier for outlier detection (typically 1.5 or 3.0)

    Returns:
        Boolean Series indicating outliers (True = outlier)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = (series < lower_bound) | (series > upper_bound)
    outlier_count = outliers.sum()
    outlier_pct = (outlier_count / len(series)) * 100

    logger.info(
        f"Detected {outlier_count} outliers ({outlier_pct:.2f}%) in {series.name}"
    )

    return outliers


def calculate_skewness(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate skewness for numerical columns.

    Args:
        df: Input DataFrame
        columns: Optional list of columns to check (defaults to all numerical)

    Returns:
        Dictionary mapping column names to skewness values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    skewness = {}
    for col in columns:
        if col in df.columns:
            skew_val = df[col].skew()
            skewness[col] = skew_val

            if abs(skew_val) > 1:
                logger.info(f"{col}: High skewness detected ({skew_val:.2f})")

    return skewness


def ensure_directory_exists(filepath: Union[str, Path]) -> None:
    """
    Ensure the directory for a given filepath exists, creating it if necessary.

    Args:
        filepath: Path to file (directory will be extracted and created)
    """
    directory = Path(filepath).parent
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Safely divide two values, handling division by zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Value to return when denominator is zero

    Returns:
        Result of division or default value
    """
    # Convert to numpy arrays for consistent handling
    num_array = np.asarray(numerator)
    denom_array = np.asarray(denominator)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom_array != 0, num_array / denom_array, default)

    # Return scalar if inputs were scalars
    if np.isscalar(numerator) and np.isscalar(denominator):
        return float(result)

    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary with configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {str(e)}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Dictionary with configuration parameters
        config_path: Path where to save JSON file
    """
    try:
        ensure_directory_exists(config_path)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced classification.

    Args:
        y: Target variable Series

    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    logger.info(f"Calculated class weights: {class_weights}")
    return class_weights


def format_large_number(num: Union[int, float], precision: int = 2) -> str:
    """
    Format large numbers with K, M, B suffixes.

    Args:
        num: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.{precision}f}B"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.{precision}f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def get_feature_importance_df(
    model, feature_names: List[str], top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained sklearn model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        top_n: Optional number of top features to return

    Returns:
        DataFrame with features and importance scores, sorted descending
    """
    try:
        # Try feature_importances_ (tree-based models)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        # Try coef_ (linear models)
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            raise AttributeError("Model doesn't have feature_importances_ or coef_")

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        logger.info(f"Extracted feature importance for {len(importance_df)} features")
        return importance_df

    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        raise


def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a feature.

    Args:
        df: DataFrame with feature and target
        feature: Name of feature column
        target: Name of binary target column (0/1)

    Returns:
        DataFrame with WoE and IV calculations per feature bin
    """
    try:
        # Create crosstab
        crosstab = pd.crosstab(df[feature], df[target], normalize="columns")

        # Calculate WoE
        crosstab["WoE"] = np.log(safe_divide(crosstab[1], crosstab[0], default=1))

        # Calculate IV
        crosstab["IV"] = (crosstab[1] - crosstab[0]) * crosstab["WoE"]

        total_iv = crosstab["IV"].sum()

        logger.info(f"Feature '{feature}' - Total IV: {total_iv:.4f}")

        # Interpret IV strength
        if total_iv < 0.02:
            strength = "Not predictive"
        elif total_iv < 0.1:
            strength = "Weak predictor"
        elif total_iv < 0.3:
            strength = "Medium predictor"
        else:
            strength = "Strong predictor"

        logger.info(f"  Predictive strength: {strength}")

        return crosstab

    except Exception as e:
        logger.error(f"Error calculating WoE/IV: {str(e)}")
        raise


def merge_transaction_features(
    transaction_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    customer_col: str = "CustomerId",
) -> pd.DataFrame:
    """
    Merge transaction-level, aggregate, and RFM features into one DataFrame.

    Args:
        transaction_df: Transaction-level features
        aggregate_df: Customer-level aggregate features
        rfm_df: RFM metrics and target variable
        customer_col: Customer identifier column

    Returns:
        Merged DataFrame ready for modeling
    """
    try:
        logger.info("Merging feature datasets")

        # Merge aggregate features
        merged = transaction_df.merge(
            aggregate_df, on=customer_col, how="left", validate="m:1"
        )

        # Merge RFM features and target
        merged = merged.merge(rfm_df, on=customer_col, how="left", validate="m:1")

        logger.info(f"Merged dataset shape: {merged.shape}")

        # Check for merge issues
        null_after_merge = merged.isnull().sum().sum()
        if null_after_merge > 0:
            logger.warning(f"Merge introduced {null_after_merge} null values")

        return merged

    except Exception as e:
        logger.error(f"Error merging features: {str(e)}")
        raise


def print_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print a comprehensive summary of a DataFrame.

    Args:
        df: DataFrame to summarize
        name: Name to display in summary
    """
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values")
    else:
        print(missing[missing > 0])
    print(f"\nNumerical Columns Summary:")
    print(df.describe())
    print(f"{'='*60}\n")


# Environment setup utilities
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        ensure_directory_exists(log_file)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logger.info(f"Logging configured at {log_level} level")


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path object pointing to project root
    """
    return Path(__file__).parent.parent


if __name__ == "__main__":
    # Example usage
    print("Utility functions module loaded successfully")
    print(f"Project root: {get_project_root()}")
