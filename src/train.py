"""
Model Training and Evaluation Module with MLflow Tracking.

This module provides comprehensive functionality for training credit risk models,
including multiple algorithms, hyperparameter tuning, evaluation metrics,
and MLflow experiment tracking.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import json

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training class with MLflow tracking.

    This class handles training multiple models, hyperparameter tuning,
    evaluation, and experiment tracking using MLflow.

    Attributes:
        experiment_name (str): MLflow experiment name
        tracking_uri (str): MLflow tracking server URI
        random_state (int): Random seed for reproducibility
        test_size (float): Proportion of data for testing
        cv_folds (int): Number of cross-validation folds
        models (Dict): Dictionary of trained models
        best_model (Any): Best performing model
        evaluation_results (Dict): Evaluation metrics for all models
    """

    def __init__(
        self,
        experiment_name: str = "credit-risk-modeling",
        tracking_uri: str = "./mlruns",
        random_state: int = 42,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: Path to MLflow tracking directory
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for testing
            cv_folds: Number of cross-validation folds
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        # Initialize storage
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.evaluation_results: Dict[str, Dict[str, float]] = {}

        # Setup MLflow
        self._setup_mlflow()

        logger.info(f"ModelTrainer initialized with experiment: {experiment_name}")

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
            logger.info(f"MLflow experiment set to: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "is_high_risk",
        feature_cols: Optional[List[str]] = None,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature columns (if None, use all except target)
            stratify: Whether to use stratified sampling

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)

        Raises:
            ValueError: If target column is missing or data is invalid
        """
        try:
            logger.info("Preparing data for training...")

            # Validate target column
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame")

            # Separate features and target
            if feature_cols is None:
                # Exclude non-feature columns
                exclude_cols = [
                    target_col,
                    "CustomerId",
                    "TransactionId",
                    "BatchId",
                    "AccountId",
                    "SubscriptionId",
                    "TransactionStartTime",
                    "Cluster",  # Keep as auxiliary but don't use for training
                ]
                feature_cols = [col for col in df.columns if col not in exclude_cols]

                # Additional check: only select numeric columns
                numeric_cols = (
                    df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
                )
                if len(numeric_cols) < len(feature_cols):
                    non_numeric = set(feature_cols) - set(numeric_cols)
                    logger.warning(f"Excluding non-numeric columns: {non_numeric}")
                feature_cols = numeric_cols

            X = df[feature_cols].copy()
            y = df[target_col].copy()

            logger.info(f"Features: {len(feature_cols)} columns")
            logger.info(f"Target: {target_col}")
            logger.info(f"Dataset shape: {X.shape}")
            logger.info(f"Target distribution:\n{y.value_counts()}")

            # Check for missing values
            if X.isnull().any().any():
                logger.warning("Missing values detected in features")
                logger.warning(
                    f"Missing values per column:\n{X.isnull().sum()[X.isnull().sum() > 0]}"
                )
                # Drop rows with missing values
                mask = ~X.isnull().any(axis=1)
                X = X[mask]
                y = y[mask]
                logger.info(f"Dropped {(~mask).sum()} rows with missing values")

            # Stratified split
            stratify_var = y if stratify else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_var,
            )

            logger.info(f"Train set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            logger.info(f"Train target distribution:\n{y_train.value_counts()}")
            logger.info(f"Test target distribution:\n{y_test.value_counts()}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        use_class_weight: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train Logistic Regression with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Hyperparameter grid for tuning
            use_class_weight: Whether to use balanced class weights

        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info("Training Logistic Regression model...")

        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "max_iter": [1000],
            }

        base_model = LogisticRegression(
            random_state=self.random_state,
            class_weight="balanced" if use_class_weight else None,
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=2
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def train_decision_tree(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        use_class_weight: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train Decision Tree with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Hyperparameter grid for tuning
            use_class_weight: Whether to use balanced class weights

        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info("Training Decision Tree model...")

        if param_grid is None:
            param_grid = {
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 4],
                "criterion": ["gini"],
            }

        base_model = DecisionTreeClassifier(
            random_state=self.random_state,
            class_weight="balanced" if use_class_weight else None,
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=2
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        use_class_weight: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train Random Forest with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Hyperparameter grid for tuning
            use_class_weight: Whether to use balanced class weights

        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info("Training Random Forest model...")

        if param_grid is None:
            param_grid = {
                "n_estimators": [100],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1],
                "max_features": ["sqrt"],
            }

        base_model = RandomForestClassifier(
            random_state=self.random_state,
            class_weight="balanced" if use_class_weight else None,
            n_jobs=-1,
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=2
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train Gradient Boosting with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Hyperparameter grid for tuning

        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info("Training Gradient Boosting model...")

        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.1, 0.3],
                "max_depth": [3, 5],
                "min_samples_split": [2],
                "min_samples_leaf": [1],
                "subsample": [0.8],
            }

        base_model = GradientBoostingClassifier(random_state=self.random_state)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=2
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance with comprehensive metrics.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }

            logger.info(f"{model_name} Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all models with MLflow tracking.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            models_to_train: List of model names to train (default: all)

        Returns:
            Dictionary containing all trained models and their metrics
        """
        if models_to_train is None:
            models_to_train = [
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "gradient_boosting",
            ]

        results = {}

        for model_name in models_to_train:
            logger.info(f"\n{'='*80}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*80}")

            with mlflow.start_run(run_name=model_name):
                try:
                    # Train model based on name
                    if model_name == "logistic_regression":
                        model, best_params = self.train_logistic_regression(
                            X_train, y_train
                        )
                    elif model_name == "decision_tree":
                        model, best_params = self.train_decision_tree(X_train, y_train)
                    elif model_name == "random_forest":
                        model, best_params = self.train_random_forest(X_train, y_train)
                    elif model_name == "gradient_boosting":
                        model, best_params = self.train_gradient_boosting(
                            X_train, y_train
                        )
                    else:
                        logger.warning(f"Unknown model: {model_name}, skipping...")
                        continue

                    # Evaluate model
                    metrics = self.evaluate_model(model, X_test, y_test, model_name)

                    # Log to MLflow
                    mlflow.log_params(best_params)
                    mlflow.log_metrics(metrics)
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("test_size", self.test_size)
                    mlflow.log_param("random_state", self.random_state)

                    # Log model
                    mlflow.sklearn.log_model(model, "model")

                    # Store results
                    self.models[model_name] = model
                    self.evaluation_results[model_name] = metrics
                    results[model_name] = {
                        "model": model,
                        "metrics": metrics,
                        "best_params": best_params,
                    }

                    logger.info(f"{model_name} training completed successfully")

                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    continue

        # Identify best model
        self._select_best_model()

        return results

    def _select_best_model(self) -> None:
        """Select the best model based on ROC-AUC score."""
        if not self.evaluation_results:
            logger.warning("No models trained yet")
            return

        best_score = -1
        best_name = None

        for model_name, metrics in self.evaluation_results.items():
            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_name = model_name

        self.best_model = self.models[best_name]
        self.best_model_name = best_name

        logger.info(f"\nBest Model: {best_name}")
        logger.info(f"ROC-AUC Score: {best_score:.4f}")

    def generate_comparison_report(
        self, save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate comparison report for all trained models.

        Args:
            save_path: Optional path to save the report

        Returns:
            DataFrame with comparison metrics
        """
        if not self.evaluation_results:
            raise ValueError("No models trained yet")

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.evaluation_results).T
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values("roc_auc", ascending=False)

        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON REPORT")
        logger.info("=" * 80)
        logger.info(f"\n{comparison_df.to_string()}")

        if save_path:
            comparison_df.to_csv(save_path)
            logger.info(f"\nComparison report saved to {save_path}")

        return comparison_df

    def plot_model_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization comparing all models.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("No models trained yet")

        # Prepare data
        models = list(self.evaluation_results.keys())
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Plot each metric
        for idx, metric in enumerate(metrics):
            values = [self.evaluation_results[model][metric] for model in models]
            axes[idx].bar(models, values, color="steelblue")
            axes[idx].set_title(
                f'{metric.replace("_", " ").title()}', fontsize=12, fontweight="bold"
            )
            axes[idx].set_ylabel("Score")
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis="x", rotation=45)

            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        # Overall comparison in last subplot
        comparison_data = []
        for model in models:
            comparison_data.append(
                [
                    self.evaluation_results[model]["accuracy"],
                    self.evaluation_results[model]["precision"],
                    self.evaluation_results[model]["recall"],
                    self.evaluation_results[model]["f1_score"],
                    self.evaluation_results[model]["roc_auc"],
                ]
            )

        comparison_df = pd.DataFrame(
            comparison_data,
            index=models,
            columns=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        )

        axes[5].axis("off")
        table = axes[5].table(
            cellText=comparison_df.round(3).values,
            rowLabels=comparison_df.index,
            colLabels=comparison_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[5].set_title("Summary Table", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()

    def save_best_model(self, output_path: str) -> None:
        """
        Save the best model to MLflow registry.

        Args:
            output_path: Path to save model artifacts
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Train models first.")

        try:
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            model_details = mlflow.register_model(
                model_uri=model_uri, name="credit-risk-classifier"
            )

            logger.info(f"Best model ({self.best_model_name}) registered in MLflow")
            logger.info(f"Model name: {model_details.name}")
            logger.info(f"Model version: {model_details.version}")

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise


def main():
    """Main execution function."""
    try:
        # Setup paths
        data_path = Path("data/processed/modeling_data.csv")
        output_dir = Path("reports/model_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("CREDIT RISK MODEL TRAINING PIPELINE")
        logger.info("=" * 80)

        # Load data
        logger.info(f"\nLoading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df):,} samples with {len(df.columns)} columns")

        # Initialize trainer
        trainer = ModelTrainer(
            experiment_name="credit-risk-modeling",
            tracking_uri="./mlruns",
            random_state=42,
            test_size=0.2,
            cv_folds=5,
        )

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            df, target_col="is_high_risk"
        )

        # Train all models
        logger.info("\nTraining all models...")
        results = trainer.train_all_models(X_train, X_test, y_train, y_test)

        # Generate reports
        logger.info("\nGenerating comparison reports...")
        comparison_df = trainer.generate_comparison_report(
            save_path=output_dir / "model_comparison.csv"
        )

        # Create visualizations
        trainer.plot_model_comparison(save_path=output_dir / "model_comparison.png")

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to {output_dir}")
        logger.info(f"MLflow tracking at: ./mlruns")
        logger.info(f"\nTo view MLflow UI, run: mlflow ui")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
