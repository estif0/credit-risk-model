"""
Unit tests for FastAPI application.

Tests the Credit Risk API endpoints, model loading, and predictions.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from src.api.main import app, MODEL_STATE, calculate_credit_score, get_confidence_level
from src.api.pydantic_models import TransactionInput, RiskPrediction


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing."""
    return {
        "TransactionId": "TXN_TEST_001",
        "AccountId": "ACC_TEST_001",
        "CustomerId": "CUST_TEST_001",
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


@pytest.fixture
def mock_model():
    """Create a mock ML model."""
    model = Mock()
    model.predict_proba.return_value = np.array(
        [[0.8, 0.2]]
    )  # Low risk (20% probability of high risk)
    return model


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_endpoint_without_model(client):
    """Test health endpoint when model is not loaded."""
    # Clear model state
    MODEL_STATE["model"] = None

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is False


def test_health_endpoint_with_model(client, mock_model):
    """Test health endpoint when model is loaded."""
    # Set model state
    MODEL_STATE["model"] = mock_model
    MODEL_STATE["model_version"] = "1.0.0"

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"


def test_model_info_without_model(client):
    """Test model info endpoint when model is not loaded."""
    MODEL_STATE["model"] = None

    response = client.get("/model/info")
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


def test_model_info_with_model(client, mock_model):
    """Test model info endpoint when model is loaded."""
    MODEL_STATE["model"] = mock_model
    MODEL_STATE["model_name"] = "test_model"
    MODEL_STATE["model_version"] = "1.0.0"
    MODEL_STATE["model_type"] = "Random Forest"
    MODEL_STATE["metrics"] = {"accuracy": 0.92, "roc_auc": 0.95}
    MODEL_STATE["loaded_at"] = datetime.utcnow()

    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test_model"
    assert data["model_version"] == "1.0.0"
    assert data["model_type"] == "Random Forest"
    assert data["accuracy"] == 0.92
    assert data["roc_auc"] == 0.95


def test_predict_without_model(client, sample_transaction):
    """Test prediction endpoint when model is not loaded."""
    MODEL_STATE["model"] = None

    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


def test_predict_with_model(client, sample_transaction, mock_model):
    """Test successful prediction."""
    MODEL_STATE["model"] = mock_model

    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200

    data = response.json()
    assert "customer_id" in data
    assert "transaction_id" in data
    assert "risk_probability" in data
    assert "risk_category" in data
    assert "credit_score" in data
    assert "confidence" in data

    assert data["customer_id"] == "CUST_TEST_001"
    assert data["transaction_id"] == "TXN_TEST_001"
    assert 0 <= data["risk_probability"] <= 1
    assert data["risk_category"] in ["low", "high"]
    assert 300 <= data["credit_score"] <= 850


def test_predict_invalid_input(client, mock_model):
    """Test prediction with invalid input."""
    MODEL_STATE["model"] = mock_model

    invalid_transaction = {
        "TransactionId": "TXN_001",
        "Amount": -100,  # Invalid: negative amount
        "Value": 5000.0,
    }

    response = client.post("/predict", json=invalid_transaction)
    assert response.status_code == 422  # Validation error


def test_batch_predict_without_model(client, sample_transaction):
    """Test batch prediction when model is not loaded."""
    MODEL_STATE["model"] = None

    batch_input = {"transactions": [sample_transaction]}
    response = client.post("/predict/batch", json=batch_input)
    assert response.status_code == 503


def test_batch_predict_with_model(client, sample_transaction, mock_model):
    """Test successful batch prediction."""
    MODEL_STATE["model"] = mock_model

    batch_input = {
        "transactions": [
            sample_transaction,
            {
                **sample_transaction,
                "TransactionId": "TXN_TEST_002",
                "CustomerId": "CUST_TEST_002",
            },
        ]
    }

    response = client.post("/predict/batch", json=batch_input)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert "total_processed" in data
    assert "model_version" in data
    assert "timestamp" in data

    assert len(data["predictions"]) == 2
    assert data["total_processed"] == 2


def test_batch_predict_empty_list(client, mock_model):
    """Test batch prediction with empty transaction list."""
    MODEL_STATE["model"] = mock_model

    batch_input = {"transactions": []}
    response = client.post("/predict/batch", json=batch_input)
    assert response.status_code == 422  # Validation error (min_length=1)


def test_calculate_credit_score():
    """Test credit score calculation."""
    # Low risk (0.1 probability) should give high score
    score_low = calculate_credit_score(0.1)
    assert 700 <= score_low <= 850

    # High risk (0.9 probability) should give low score
    score_high = calculate_credit_score(0.9)
    assert 300 <= score_high <= 500

    # Medium risk (0.5 probability) should give medium score
    score_medium = calculate_credit_score(0.5)
    assert 500 <= score_medium <= 650

    # Boundary cases
    assert calculate_credit_score(0.0) == 850
    assert calculate_credit_score(1.0) == 300


def test_get_confidence_level():
    """Test confidence level determination."""
    # High confidence for extreme probabilities
    assert get_confidence_level(0.1) == "high"
    assert get_confidence_level(0.9) == "high"

    # Medium confidence
    assert get_confidence_level(0.35) == "medium"
    assert get_confidence_level(0.65) == "medium"

    # Low confidence for probabilities near 0.5
    assert get_confidence_level(0.5) == "low"


def test_predict_high_risk_customer(client, mock_model):
    """Test prediction for high-risk customer."""
    # Mock model to return high risk probability
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% high risk
    MODEL_STATE["model"] = mock_model

    transaction = {
        "TransactionId": "TXN_HIGHRISK_001",
        "AccountId": "ACC_001",
        "CustomerId": "CUST_HIGHRISK_001",
        "Amount": 100.0,
        "Value": 100.0,
        "transaction_hour": 3,
        "transaction_day": 1,
        "transaction_month": 1,
        "transaction_year": 2024,
        "is_weekend": 0,
        "total_transaction_value": 500.0,
        "avg_transaction_value": 100.0,
        "transaction_count": 5,
        "Recency": 90,  # Long time since last transaction
        "Frequency": 5,  # Low frequency
        "Monetary": 500.0,  # Low monetary value
    }

    response = client.post("/predict", json=transaction)
    assert response.status_code == 200

    data = response.json()
    assert data["risk_category"] == "high"
    assert data["risk_probability"] > 0.5
    assert data["credit_score"] < 550  # Low credit score for high risk


def test_predict_low_risk_customer(client, mock_model):
    """Test prediction for low-risk customer."""
    # Mock model to return low risk probability
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])  # 10% high risk
    MODEL_STATE["model"] = mock_model

    transaction = {
        "TransactionId": "TXN_LOWRISK_001",
        "AccountId": "ACC_001",
        "CustomerId": "CUST_LOWRISK_001",
        "Amount": 10000.0,
        "Value": 10000.0,
        "transaction_hour": 14,
        "transaction_day": 15,
        "transaction_month": 12,
        "transaction_year": 2024,
        "is_weekend": 0,
        "total_transaction_value": 500000.0,
        "avg_transaction_value": 10000.0,
        "transaction_count": 50,
        "Recency": 2,  # Recent transaction
        "Frequency": 50,  # High frequency
        "Monetary": 500000.0,  # High monetary value
    }

    response = client.post("/predict", json=transaction)
    assert response.status_code == 200

    data = response.json()
    assert data["risk_category"] == "low"
    assert data["risk_probability"] < 0.5
    assert data["credit_score"] > 700  # High credit score for low risk


@pytest.mark.parametrize(
    "hour,expected_valid", [(0, True), (12, True), (23, True), (24, False), (-1, False)]
)
def test_transaction_hour_validation(client, sample_transaction, hour, expected_valid):
    """Test transaction hour validation."""
    sample_transaction["transaction_hour"] = hour

    response = client.post("/predict", json=sample_transaction)

    if expected_valid:
        assert response.status_code in [200, 503]  # 503 if model not loaded
    else:
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
