"""
Credit Risk API - FastAPI Application.

This module provides a REST API for credit risk prediction using trained
machine learning models stored in MLflow.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware import RequestLoggingMiddleware, RateLimitMiddleware

from src.api.pydantic_models import (
    TransactionInput,
    BatchPredictionInput,
    RiskPrediction,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    ModelInfo,
)
from src.config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.api.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add Middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, limit=100, window=60)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL_STATE = {
    "model": None,
    "model_name": None,
    "model_version": None,
    "model_type": None,
    "features": None,
    "metrics": {},
    "loaded_at": None,
}


class ModelManager:
    """Manager for loading and accessing ML models."""

    def __init__(self):
        self.mlflow_tracking_uri = config.ml.tracking_uri
        self.experiment_name = config.ml.experiment_name
        self.model_stage = config.ml.model_stage

    def load_best_model(self):
        """
        Load the best model from MLflow experiment.

        Returns:
            Loaded model and metadata
        """
        try:
            logger.info(
                f"Loading model from MLflow tracking URI: {self.mlflow_tracking_uri}"
            )
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            # Get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{self.experiment_name}' not found")

            # Search for best run (highest ROC-AUC)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["metrics.roc_auc DESC"],
                max_results=3,
            )

            if runs.empty:
                raise ValueError("No finished runs found in experiment")

            best_run = None
            run_id = None
            model = None

            for _, run in runs.iterrows():
                try:
                    current_run_id = run.run_id
                    logger.info(f"Attempting to load model from run: {current_run_id}")
                    model_uri = f"runs:/{current_run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    best_run = run
                    run_id = current_run_id
                    logger.info(f"Successfully loaded model from run: {run_id}")
                    break
                except Exception as e:
                    logger.warning(
                        f"Failed to load model from run {current_run_id}: {e}"
                    )
                    continue

            if model is None:
                raise RuntimeError("Failed to load any model from top runs")

            # Extract metadata
            metrics = {
                "accuracy": best_run.get("metrics.accuracy"),
                "precision": best_run.get("metrics.precision"),
                "recall": best_run.get("metrics.recall"),
                "f1_score": best_run.get("metrics.f1_score"),
                "roc_auc": best_run.get("metrics.roc_auc"),
            }

            model_name = best_run.get("tags.mlflow.runName", "unknown")

            # Update global state
            self._update_global_state(model, best_run, model_name, metrics)

            return model, best_run

        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            raise

    def list_models(self) -> List[Dict]:
        """List all available models (runs) from MLflow."""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is None:
                return []

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
            )

            model_list = []
            for _, run in runs.iterrows():
                # Helper to safely get metric
                def get_metric(metric_name):
                    val = run.get(f"metrics.{metric_name}")
                    if val is None:
                        return None
                    try:
                        fval = float(val)
                        if pd.isna(fval) or np.isinf(fval):
                            return None
                        return fval
                    except (ValueError, TypeError):
                        return None

                model_list.append(
                    {
                        "run_id": run.run_id,
                        "name": run.get("tags.mlflow.runName", "unknown"),
                        "type": run.get("params.model_type", "unknown"),
                        "roc_auc": get_metric("roc_auc"),
                        "accuracy": get_metric("accuracy"),
                        "created": pd.to_datetime(run.start_time).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )

            return model_list

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def load_model_by_run_id(self, run_id: str):
        """Load a specific model by run_id."""
        try:
            logger.info(f"Loading model from run: {run_id}")
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            run = mlflow.get_run(run_id)

            # Extract metadata
            metrics = {
                "accuracy": run.data.metrics.get("accuracy"),
                "roc_auc": run.data.metrics.get("roc_auc"),
            }

            model_name = run.data.tags.get("mlflow.runName", "unknown")

            # Pass the full Run object
            self._update_global_state(model, run, model_name, metrics)
            return model, run

        except Exception as e:
            logger.error(f"Error loading model {run_id}: {e}")
            raise

    def _update_global_state(self, model, run_data, model_name, metrics):
        """Helper to update global model state."""
        MODEL_STATE["model"] = model
        MODEL_STATE["model_name"] = model_name

        # run_data can be a pandas Series (from search_runs) or mlflow.entities.Run (from get_run)
        if isinstance(run_data, pd.Series):
            MODEL_STATE["model_version"] = run_data.run_id
            # Fix: params keys in search_runs are prefixed with 'params.'
            model_type = run_data.get("params.model_type")
            if not model_type or pd.isna(model_type):
                # Fallback: try to infer from run name or tags
                model_type = run_data.get("tags.mlflow.runName", "Unknown")
            MODEL_STATE["model_type"] = model_type

            # Extract start_time
            start_time = run_data.get("start_time")
            if start_time:
                MODEL_STATE["created"] = pd.to_datetime(start_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        else:
            # Assume mlflow.entities.Run
            MODEL_STATE["model_version"] = run_data.info.run_id
            model_type = run_data.data.params.get("model_type")
            if not model_type:
                # Fallback
                model_type = run_data.data.tags.get("mlflow.runName", "Unknown")
            MODEL_STATE["model_type"] = model_type

            # Extract start_time
            if run_data.info.start_time:
                # MLflow stores time in ms? verify. usually it's timestamp in ms
                # pd.to_datetime handles int/float as ns by default, or verify conversion
                try:
                    dt = datetime.fromtimestamp(run_data.info.start_time / 1000.0)
                    MODEL_STATE["created"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    MODEL_STATE["created"] = "Unknown"

        MODEL_STATE["metrics"] = metrics
        MODEL_STATE["loaded_at"] = datetime.utcnow()


# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    try:
        logger.info("Starting Credit Risk API...")
        model_manager.load_best_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning("API will start without a loaded model")


@app.get("/model/list", tags=["Model Management"])
async def list_models():
    """List all available trained models."""
    models = model_manager.list_models()
    return {"models": models}


@app.post("/model/load/{run_id}", tags=["Model Management"])
async def load_model(run_id: str):
    """Load a specific model by run ID."""
    try:
        model, run = model_manager.load_model_by_run_id(run_id)

        return {
            "message": f"Successfully loaded model from run {run_id}",
            "model_info": {
                "name": MODEL_STATE["model_name"],
                "type": MODEL_STATE["model_type"],
                "version": MODEL_STATE["model_version"],
                "metrics": MODEL_STATE["metrics"],
                "loaded_at": MODEL_STATE["loaded_at"],
                "created": MODEL_STATE.get("created", "Unknown"),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load model: {str(e)}",
        )


@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint."""
    return {
        "message": "Welcome to the Credit Risk Scoring API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "model_list": "/model/list",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse: Service health status
    """
    model_loaded = MODEL_STATE["model"] is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=MODEL_STATE.get("model_version"),
        timestamp=datetime.utcnow(),
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        ModelInfo: Model metadata and metrics

    Raises:
        HTTPException: If model is not loaded
    """
    if MODEL_STATE["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator.",
        )

    metrics = MODEL_STATE.get("metrics", {})

    return ModelInfo(
        model_name=MODEL_STATE.get("model_name", "unknown"),
        model_version=MODEL_STATE.get("model_version", "unknown"),
        model_type=MODEL_STATE.get("model_type", "unknown"),
        accuracy=metrics.get("accuracy"),
        roc_auc=metrics.get("roc_auc"),
        f1_score=metrics.get("f1_score"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        training_date=MODEL_STATE.get("created", "Unknown"),
        loaded_at=MODEL_STATE.get("loaded_at"),
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """
    Reload the model from MLflow.

    Returns:
        dict: Reload status
    """
    try:
        model_manager.load_best_model()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_name": MODEL_STATE.get("model_name"),
            "timestamp": datetime.utcnow(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}",
        )


def calculate_credit_score(risk_probability: float) -> int:
    """
    Convert risk probability to credit score (300-850 scale).

    Args:
        risk_probability: Risk probability (0-1)

    Returns:
        Credit score (300-850)
    """
    # Inverse relationship: low risk = high score
    # Map 0-1 probability to 850-300 score
    score = int(850 - (risk_probability * 550))
    return max(300, min(850, score))


def get_confidence_level(risk_probability: float) -> str:
    """
    Determine confidence level based on probability.

    Args:
        risk_probability: Risk probability (0-1)

    Returns:
        Confidence level string
    """
    if risk_probability < 0.3 or risk_probability > 0.7:
        return "high"
    elif risk_probability < 0.4 or risk_probability > 0.6:
        return "medium"
    else:
        return "low"


def prepare_features(transaction: TransactionInput) -> pd.DataFrame:
    """
    Prepare features from transaction input for model prediction.
    """
    # Aggregates and advanced features
    avg_val = transaction.avg_transaction_value
    std_val = transaction.std_transaction_value or 0.0
    min_val = (
        transaction.min_transaction_value
        if transaction.min_transaction_value is not None
        else transaction.Amount
    )
    max_val = (
        transaction.max_transaction_value
        if transaction.max_transaction_value is not None
        else transaction.Amount
    )

    value_range = max_val - min_val
    value_cv = std_val / avg_val if avg_val > 0 else 0.0

    features = {
        "CountryCode": transaction.CountryCode or 256,
        "Amount": transaction.Amount,
        "Value": transaction.Value,
        "PricingStrategy": transaction.PricingStrategy or 2,
        "FraudResult": transaction.FraudResult or 0,
        "transaction_hour": transaction.transaction_hour,
        "transaction_day": transaction.transaction_day,
        "transaction_month": transaction.transaction_month,
        "transaction_year": transaction.transaction_year,
        "is_weekend": transaction.is_weekend,
        "total_transaction_value": transaction.total_transaction_value,
        "avg_transaction_value": avg_val,
        "std_transaction_value": std_val,
        "min_transaction_value": min_val,
        "max_transaction_value": max_val,
        "transaction_count": transaction.transaction_count,
        "value_range": value_range,
        "value_cv": value_cv,
        "ProductCategory_woe": transaction.ProductCategory_woe or 0.0,
        "ChannelId_woe": transaction.ChannelId_woe or 0.0,
        "Recency": transaction.Recency,
        "Frequency": transaction.Frequency,
        "Monetary": transaction.Monetary,
    }

    # Create DataFrame with specific column order
    column_order = [
        "CountryCode",
        "Amount",
        "Value",
        "PricingStrategy",
        "FraudResult",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
        "is_weekend",
        "total_transaction_value",
        "avg_transaction_value",
        "std_transaction_value",
        "min_transaction_value",
        "max_transaction_value",
        "transaction_count",
        "value_range",
        "value_cv",
        "ProductCategory_woe",
        "ChannelId_woe",
        "Recency",
        "Frequency",
        "Monetary",
    ]

    df = pd.DataFrame([features])[column_order]

    return df


@app.post("/predict", response_model=RiskPrediction, tags=["Prediction"])
async def predict_risk(transaction: TransactionInput):
    """
    Predict credit risk for a single transaction.

    Args:
        transaction: Transaction input data

    Returns:
        RiskPrediction: Risk prediction with probability and score

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if MODEL_STATE["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator.",
        )

    try:
        # Prepare features
        features_df = prepare_features(transaction)

        # Make prediction
        model = MODEL_STATE["model"]
        risk_prob = model.predict_proba(features_df)[0][
            1
        ]  # Probability of class 1 (high risk)

        # Calculate credit score and confidence
        credit_score = calculate_credit_score(risk_prob)
        confidence = get_confidence_level(risk_prob)
        risk_category = "high" if risk_prob > 0.5 else "low"

        logger.info(
            f"Prediction for {transaction.CustomerId}: "
            f"risk={risk_prob:.4f}, score={credit_score}, category={risk_category}"
        )

        return RiskPrediction(
            customer_id=transaction.CustomerId,
            transaction_id=transaction.TransactionId,
            risk_probability=round(risk_prob, 4),
            risk_category=risk_category,
            credit_score=credit_score,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Predict credit risk for multiple transactions.

    Args:
        batch_input: Batch of transaction inputs

    Returns:
        BatchPredictionResponse: List of risk predictions

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if MODEL_STATE["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator.",
        )

    try:
        predictions = []

        for transaction in batch_input.transactions:
            # Prepare features
            features_df = prepare_features(transaction)

            # Make prediction
            model = MODEL_STATE["model"]
            risk_prob = model.predict_proba(features_df)[0][1]

            # Calculate credit score and confidence
            credit_score = calculate_credit_score(risk_prob)
            confidence = get_confidence_level(risk_prob)
            risk_category = "high" if risk_prob > 0.5 else "low"

            predictions.append(
                RiskPrediction(
                    customer_id=transaction.CustomerId,
                    transaction_id=transaction.TransactionId,
                    risk_probability=round(risk_prob, 4),
                    risk_category=risk_category,
                    credit_score=credit_score,
                    confidence=confidence,
                )
            )

        logger.info(
            f"Batch prediction completed: {len(predictions)} transactions processed"
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            model_version=MODEL_STATE.get("model_version", "unknown"),
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    # Run the API
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
        log_level=config.api.log_level.lower(),
    )
