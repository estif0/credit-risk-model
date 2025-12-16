"""
Pydantic Models for Credit Risk API.

This module defines request and response schemas for the FastAPI endpoints,
providing data validation and serialization.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator
from datetime import datetime


class TransactionInput(BaseModel):
    """
    Input schema for a single transaction prediction request.

    Attributes:
        TransactionId: Unique transaction identifier
        AccountId: Account identifier
        CustomerId: Customer identifier
        Amount: Transaction amount
        Value: Transaction value
        transaction_hour: Hour of the day (0-23)
        transaction_day: Day of the month (1-31)
        transaction_month: Month of the year (1-12)
        transaction_year: Year
        is_weekend: Whether transaction occurred on weekend (0 or 1)
        total_transaction_value: Total transaction value for customer
        avg_transaction_value: Average transaction value for customer
        transaction_count: Number of transactions for customer
        Recency: Days since last transaction
        Frequency: Number of transactions
        Monetary: Total monetary value
    """

    TransactionId: str = Field(..., description="Unique transaction identifier")
    AccountId: str = Field(..., description="Account identifier")
    CustomerId: str = Field(..., description="Customer identifier")
    Amount: float = Field(..., gt=0, description="Transaction amount")
    Value: float = Field(..., description="Transaction value")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of day")
    transaction_day: int = Field(..., ge=1, le=31, description="Day of month")
    transaction_month: int = Field(..., ge=1, le=12, description="Month of year")
    transaction_year: int = Field(..., ge=2000, le=2100, description="Year")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend indicator")
    total_transaction_value: float = Field(
        ..., description="Total customer transaction value"
    )
    avg_transaction_value: float = Field(
        ..., gt=0, description="Average customer transaction value"
    )
    transaction_count: int = Field(
        ..., gt=0, description="Number of customer transactions"
    )
    Recency: int = Field(..., ge=0, description="Days since last transaction")
    Frequency: int = Field(..., gt=0, description="Transaction frequency")
    Monetary: float = Field(..., description="Total monetary value")

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionId": "TXN_12345",
                "AccountId": "ACC_001",
                "CustomerId": "CUST_001",
                "Amount": 5000.0,
                "Value": 5000.0,
                "transaction_hour": 14,
                "transaction_day": 15,
                "transaction_month": 12,
                "transaction_year": 2024,
                "is_weekend": 0,
                "total_transaction_value": 150000.0,
                "avg_transaction_value": 5000.0,
                "transaction_count": 30,
                "Recency": 5,
                "Frequency": 30,
                "Monetary": 150000.0,
            }
        }


class BatchPredictionInput(BaseModel):
    """
    Input schema for batch prediction requests.

    Attributes:
        transactions: List of transactions to predict
    """

    transactions: List[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions (max 1000)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "TransactionId": "TXN_001",
                        "AccountId": "ACC_001",
                        "CustomerId": "CUST_001",
                        "Amount": 5000.0,
                        "Value": 5000.0,
                        "transaction_hour": 14,
                        "transaction_day": 15,
                        "transaction_month": 12,
                        "transaction_year": 2024,
                        "is_weekend": 0,
                        "total_transaction_value": 150000.0,
                        "avg_transaction_value": 5000.0,
                        "transaction_count": 30,
                        "Recency": 5,
                        "Frequency": 30,
                        "Monetary": 150000.0,
                    }
                ]
            }
        }


class RiskPrediction(BaseModel):
    """
    Output schema for risk prediction response.

    Attributes:
        customer_id: Customer identifier
        transaction_id: Transaction identifier
        risk_probability: Probability of being high risk (0-1)
        risk_category: Risk category ("low" or "high")
        credit_score: Estimated credit score (300-850)
        confidence: Model confidence level
    """

    customer_id: str = Field(..., description="Customer identifier")
    transaction_id: str = Field(..., description="Transaction identifier")
    risk_probability: float = Field(..., ge=0, le=1, description="Risk probability")
    risk_category: str = Field(..., description="Risk category (low/high)")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "transaction_id": "TXN_001",
                "risk_probability": 0.15,
                "risk_category": "low",
                "credit_score": 720,
                "confidence": "high",
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Output schema for batch prediction response.

    Attributes:
        predictions: List of risk predictions
        total_processed: Total number of predictions
        model_version: Version of the model used
        timestamp: Prediction timestamp
    """

    predictions: List[RiskPrediction] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total predictions made")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Prediction timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": "CUST_001",
                        "transaction_id": "TXN_001",
                        "risk_probability": 0.15,
                        "risk_category": "low",
                        "credit_score": 720,
                        "confidence": "high",
                    }
                ],
                "total_processed": 1,
                "model_version": "1.0.0",
                "timestamp": "2024-12-16T10:30:00Z",
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response schema.

    Attributes:
        status: Service status
        model_loaded: Whether model is loaded
        model_version: Version of loaded model
        timestamp: Check timestamp
    """

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model loaded status")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Check timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0",
                "timestamp": "2024-12-16T10:30:00Z",
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response schema.

    Attributes:
        error: Error type
        message: Error message
        detail: Detailed error information
    """

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "detail": "Amount must be greater than 0",
            }
        }


class ModelInfo(BaseModel):
    """
    Model information response schema.

    Attributes:
        model_name: Name of the model
        model_version: Version of the model
        model_type: Type of model (e.g., "Logistic Regression")
        accuracy: Model accuracy
        roc_auc: Model ROC-AUC score
        training_date: Date model was trained
        features: List of feature names
    """

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Model accuracy")
    roc_auc: Optional[float] = Field(None, ge=0, le=1, description="ROC-AUC score")
    training_date: Optional[str] = Field(None, description="Training date")
    features: Optional[List[str]] = Field(None, description="Feature names")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "credit-risk-classifier",
                "model_version": "1.0.0",
                "model_type": "Random Forest",
                "accuracy": 0.92,
                "roc_auc": 0.95,
                "training_date": "2024-12-16",
                "features": ["Recency", "Frequency", "Monetary", "transaction_count"],
            }
        }
