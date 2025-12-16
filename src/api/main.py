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

from src.api.pydantic_models import (
    TransactionInput,
    BatchPredictionInput,
    RiskPrediction,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    ModelInfo,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using transaction data and RFM metrics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

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
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        self.experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "credit-risk-modeling"
        )
        self.model_stage = os.getenv("MODEL_STAGE", "Production")

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
                max_results=1,
            )

            if runs.empty:
                raise ValueError("No finished runs found in experiment")

            best_run = runs.iloc[0]
            run_id = best_run.run_id

            logger.info(f"Loading model from run: {run_id}")

            # Load model
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

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
            MODEL_STATE["model"] = model
            MODEL_STATE["model_name"] = model_name
            MODEL_STATE["model_version"] = "1.0.0"
            MODEL_STATE["model_type"] = best_run.get("params.model_type", "unknown")
            MODEL_STATE["metrics"] = metrics
            MODEL_STATE["loaded_at"] = datetime.utcnow()

            logger.info(f"Model loaded successfully: {model_name}")
            logger.info(f"Model metrics: {metrics}")

            return model, metrics

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


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


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Credit Risk Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
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
        training_date=(
            MODEL_STATE.get("loaded_at").strftime("%Y-%m-%d")
            if MODEL_STATE.get("loaded_at")
            else None
        ),
        features=MODEL_STATE.get("features"),
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

    Args:
        transaction: Transaction input data

    Returns:
        DataFrame with features
    """
    # Create feature dictionary (excluding IDs)
    features = {
        "Amount": transaction.Amount,
        "Value": transaction.Value,
        "transaction_hour": transaction.transaction_hour,
        "transaction_day": transaction.transaction_day,
        "transaction_month": transaction.transaction_month,
        "transaction_year": transaction.transaction_year,
        "is_weekend": transaction.is_weekend,
        "total_transaction_value": transaction.total_transaction_value,
        "avg_transaction_value": transaction.avg_transaction_value,
        "transaction_count": transaction.transaction_count,
        "Recency": transaction.Recency,
        "Frequency": transaction.Frequency,
        "Monetary": transaction.Monetary,
    }

    # Convert to DataFrame
    df = pd.DataFrame([features])

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
        "src.api.main:app", host="0.0.0.0", port=9000, reload=True, log_level="info"
    )
