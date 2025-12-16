"""
Unit tests for train module.

Tests the ModelTrainer class and model training functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import mlflow

from src.train import ModelTrainer


@pytest.fixture
def sample_train_data():
    """Fixture providing sample training data."""
    np.random.seed(42)

    n_samples = 1000

    # Create features
    df = pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.exponential(1, n_samples),
            "feature_4": np.random.randint(0, 5, n_samples),
            "is_high_risk": np.random.binomial(1, 0.3, n_samples),  # Target
            "CustomerId": [
                f"CUST_{i:04d}" for i in range(n_samples)
            ],  # Should be excluded
            "TransactionId": [
                f"TXN_{i:04d}" for i in range(n_samples)
            ],  # Should be excluded
        }
    )

    return df


@pytest.fixture
def trainer():
    """Fixture providing ModelTrainer instance."""
    return ModelTrainer(
        experiment_name="test-experiment",
        tracking_uri="./test_mlruns",
        random_state=42,
        test_size=0.2,
        cv_folds=3,
    )


def test_model_trainer_initialization(trainer):
    """Test ModelTrainer initialization."""
    assert trainer.experiment_name == "test-experiment"
    assert trainer.tracking_uri == "./test_mlruns"
    assert trainer.random_state == 42
    assert trainer.test_size == 0.2
    assert trainer.cv_folds == 3
    assert isinstance(trainer.models, dict)
    assert isinstance(trainer.evaluation_results, dict)


def test_prepare_data_basic(trainer, sample_train_data):
    """Test basic data preparation."""
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        sample_train_data, target_col="is_high_risk"
    )

    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_train_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check test size ratio
    assert abs(len(X_test) / len(sample_train_data) - 0.2) < 0.05

    # Check that ID columns are excluded
    assert "CustomerId" not in X_train.columns
    assert "TransactionId" not in X_train.columns
    assert "is_high_risk" not in X_train.columns


def test_prepare_data_stratified(trainer, sample_train_data):
    """Test stratified sampling in data preparation."""
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        sample_train_data, target_col="is_high_risk", stratify=True
    )

    # Check that class proportions are similar in train and test
    train_proportion = y_train.mean()
    test_proportion = y_test.mean()

    assert abs(train_proportion - test_proportion) < 0.1


def test_prepare_data_missing_target(trainer, sample_train_data):
    """Test error handling for missing target column."""
    with pytest.raises(ValueError, match="Target column.*not found"):
        trainer.prepare_data(sample_train_data, target_col="nonexistent_column")


def test_prepare_data_with_missing_values(trainer):
    """Test data preparation with missing values."""
    df = pd.DataFrame(
        {
            "feature_1": [1, 2, np.nan, 4, 5, 6, 7, 8],
            "feature_2": [5, np.nan, 3, 2, 1, 6, 7, 8],
            "is_high_risk": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    X_train, X_test, y_train, y_test = trainer.prepare_data(df, stratify=False)

    # Should drop rows with missing values
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0


def test_train_logistic_regression(trainer, sample_train_data):
    """Test Logistic Regression training."""
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_train_data)

    # Use small param grid for speed
    param_grid = {
        "C": [0.1, 1.0],
        "penalty": ["l2"],
        "solver": ["liblinear"],
        "max_iter": [100],
    }

    model, best_params = trainer.train_logistic_regression(
        X_train, y_train, param_grid=param_grid
    )

    assert model is not None
    assert isinstance(best_params, dict)
    assert "C" in best_params

    # Check that model can predict
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_train_decision_tree(trainer, sample_train_data):
    """Test Decision Tree training."""
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_train_data)

    # Use small param grid for speed
    param_grid = {
        "max_depth": [3, 5],
        "min_samples_split": [2, 5],
        "criterion": ["gini"],
    }

    model, best_params = trainer.train_decision_tree(
        X_train, y_train, param_grid=param_grid
    )

    assert model is not None
    assert isinstance(best_params, dict)
    assert "max_depth" in best_params

    # Check that model can predict
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)


def test_train_random_forest(trainer, sample_train_data):
    """Test Random Forest training."""
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_train_data)

    # Use small param grid for speed
    param_grid = {
        "n_estimators": [10, 50],
        "max_depth": [3, 5],
        "min_samples_split": [2],
        "max_features": ["sqrt"],
    }

    model, best_params = trainer.train_random_forest(
        X_train, y_train, param_grid=param_grid
    )

    assert model is not None
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params


def test_train_gradient_boosting(trainer, sample_train_data):
    """Test Gradient Boosting training."""
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_train_data)

    # Use small param grid for speed
    param_grid = {
        "n_estimators": [10, 50],
        "learning_rate": [0.1],
        "max_depth": [3],
        "subsample": [0.8],
    }

    model, best_params = trainer.train_gradient_boosting(
        X_train, y_train, param_grid=param_grid
    )

    assert model is not None
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params


def test_evaluate_model(trainer, sample_train_data):
    """Test model evaluation."""
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_train_data)

    # Train a simple model
    param_grid = {
        "C": [1.0],
        "penalty": ["l2"],
        "solver": ["liblinear"],
        "max_iter": [100],
    }
    model, _ = trainer.train_logistic_regression(
        X_train, y_train, param_grid=param_grid
    )

    # Evaluate
    metrics = trainer.evaluate_model(model, X_test, y_test, "test_model")

    # Check all metrics are present
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics

    # Check metric values are valid
    for metric, value in metrics.items():
        assert 0 <= value <= 1, f"{metric} should be between 0 and 1"


@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metrics")
@patch("mlflow.sklearn.log_model")
def test_train_all_models(
    mock_log_model,
    mock_log_metrics,
    mock_log_params,
    mock_start_run,
    trainer,
    sample_train_data,
):
    """Test training all models with MLflow tracking."""
    # Mock MLflow run context
    mock_run = MagicMock()
    mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
    mock_start_run.return_value.__exit__ = Mock(return_value=None)

    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_train_data)

    # Train only 1 model for speed
    results = trainer.train_all_models(
        X_train,
        X_test,
        y_train,
        y_test,
        models_to_train=["logistic_regression"],
    )

    # Check results structure
    assert len(results) >= 1
    assert "logistic_regression" in results

    for model_name, result in results.items():
        assert "model" in result
        assert "metrics" in result
        assert "best_params" in result

        # Check metrics
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "roc_auc" in metrics

    # Verify MLflow was called
    assert mock_log_params.called
    assert mock_log_metrics.called


def test_select_best_model(trainer, sample_train_data):
    """Test best model selection."""
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_train_data)

    # Manually add some evaluation results
    trainer.models["model_a"] = Mock()
    trainer.models["model_b"] = Mock()
    trainer.evaluation_results = {
        "model_a": {"roc_auc": 0.75, "accuracy": 0.70},
        "model_b": {"roc_auc": 0.85, "accuracy": 0.80},
    }

    trainer._select_best_model()

    assert trainer.best_model_name == "model_b"
    assert trainer.best_model == trainer.models["model_b"]


def test_generate_comparison_report(trainer):
    """Test comparison report generation."""
    # Add mock evaluation results
    trainer.evaluation_results = {
        "model_1": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.80,
            "f1_score": 0.81,
            "roc_auc": 0.88,
        },
        "model_2": {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.85,
            "f1_score": 0.86,
            "roc_auc": 0.92,
        },
    }

    comparison_df = trainer.generate_comparison_report()

    # Check DataFrame structure
    assert len(comparison_df) == 2
    assert "accuracy" in comparison_df.columns
    assert "roc_auc" in comparison_df.columns

    # Check sorting (should be sorted by roc_auc descending)
    assert comparison_df.index[0] == "model_2"


def test_generate_comparison_report_no_models(trainer):
    """Test comparison report fails with no trained models."""
    with pytest.raises(ValueError, match="No models trained yet"):
        trainer.generate_comparison_report()


def test_deterministic_results(sample_train_data):
    """Test that training produces deterministic results with same random_state."""
    # Train twice with same random state
    trainer1 = ModelTrainer(random_state=42, test_size=0.2, cv_folds=2)
    X_train1, X_test1, y_train1, y_test1 = trainer1.prepare_data(sample_train_data)

    trainer2 = ModelTrainer(random_state=42, test_size=0.2, cv_folds=2)
    X_train2, X_test2, y_train2, y_test2 = trainer2.prepare_data(sample_train_data)

    # Results should be identical
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
