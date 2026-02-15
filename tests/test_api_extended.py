"""
Additional unit tests for FastAPI application - Model Management & Edge Cases.

Tests the model management endpoints and edge cases not covered in test_api.py.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from src.api.main import app, MODEL_STATE
from src.api.pydantic_models import TransactionInput


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
    model.predict_proba.return_value = np.array([[0.8, 0.2]])
    return model


# ============================================================================
# Model Management Endpoints Tests
# ============================================================================


def test_list_models_endpoint(client):
    """Test the /model/list endpoint."""
    with patch("src.api.main.model_manager.list_models") as mock_list:
        mock_list.return_value = [
            {
                "run_id": "abc123",
                "model_name": "test_model_1",
                "model_type": "gradient_boosting",
                "accuracy": 0.92,
                "roc_auc": 0.95,
            },
            {
                "run_id": "def456",
                "model_name": "test_model_2",
                "model_type": "random_forest",
                "accuracy": 0.90,
                "roc_auc": 0.93,
            },
        ]

        response = client.get("/model/list")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 2
        assert data["models"][0]["run_id"] == "abc123"


def test_list_models_empty(client):
    """Test /model/list when no models are available."""
    with patch("src.api.main.model_manager.list_models") as mock_list:
        mock_list.return_value = []

        response = client.get("/model/list")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []


def test_load_model_by_run_id_success(client, mock_model):
    """Test successfully loading a model by run ID."""
    mock_run = Mock()
    mock_run.info.run_id = "test_run_123"

    with patch("src.api.main.model_manager.load_model_by_run_id") as mock_load:
        mock_load.return_value = (mock_model, mock_run)
        MODEL_STATE["model_name"] = "test_model"
        MODEL_STATE["model_type"] = "gradient_boosting"
        MODEL_STATE["model_version"] = "test_run_123"
        MODEL_STATE["metrics"] = {"accuracy": 0.92}
        MODEL_STATE["loaded_at"] = datetime.utcnow()

        response = client.post("/model/load/test_run_123")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "model_info" in data
        assert data["model_info"]["version"] == "test_run_123"


def test_load_model_by_run_id_failure(client):
    """Test loading a model with invalid run ID."""
    with patch("src.api.main.model_manager.load_model_by_run_id") as mock_load:
        mock_load.side_effect = Exception("Run not found")

        response = client.post("/model/load/invalid_run_id")
        assert response.status_code == 400
        assert "Failed to load model" in response.json()["detail"]


def test_model_reload_success(client, mock_model):
    """Test the /model/reload endpoint."""
    with patch("src.api.main.model_manager.load_best_model") as mock_reload:
        mock_reload.return_value = (mock_model, Mock())
        MODEL_STATE["model_name"] = "reloaded_model"

        response = client.post("/model/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "timestamp" in data


def test_model_reload_failure(client):
    """Test /model/reload when reload fails."""
    with patch("src.api.main.model_manager.load_best_model") as mock_reload:
        mock_reload.side_effect = Exception("MLflow connection error")

        response = client.post("/model/reload")
        assert response.status_code == 500
        assert "Failed to reload model" in response.json()["detail"]


# ============================================================================
# Edge Cases & Error Handling Tests
# ============================================================================


def test_predict_model_prediction_error(client, sample_transaction, mock_model):
    """Test prediction when model.predict_proba raises an error."""
    MODEL_STATE["model"] = mock_model
    mock_model.predict_proba.side_effect = ValueError("Invalid feature values")

    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]


def test_batch_predict_max_limit(client, sample_transaction, mock_model):
    """Test batch prediction with exactly 1000 transactions (max limit)."""
    MODEL_STATE["model"] = mock_model

    # Create 1000 transactions
    transactions = [
        {**sample_transaction, "TransactionId": f"TXN_{i:04d}"} for i in range(1000)
    ]
    batch_input = {"transactions": transactions}

    response = client.post("/predict/batch", json=batch_input)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 1000
    assert len(data["predictions"]) == 1000


def test_batch_predict_exceeds_max_limit(client, sample_transaction):
    """Test batch prediction with more than 1000 transactions (should fail)."""
    # Create 1001 transactions
    transactions = [
        {**sample_transaction, "TransactionId": f"TXN_{i:04d}"} for i in range(1001)
    ]
    batch_input = {"transactions": transactions}

    response = client.post("/predict/batch", json=batch_input)
    assert response.status_code == 422  # Validation error


# ============================================================================
# Input Validation Tests
# ============================================================================


@pytest.mark.parametrize(
    "field,value,should_pass",
    [
        ("Amount", 0.01, True),  # Minimum valid amount
        ("Amount", 0, False),  # Zero amount (gt=0 constraint)
        ("Amount", -100, False),  # Negative amount
        ("transaction_year", 2000, True),  # Minimum year
        ("transaction_year", 2100, True),  # Maximum year
        ("transaction_year", 1999, False),  # Below minimum
        ("transaction_year", 2101, False),  # Above maximum
        ("transaction_month", 1, True),  # Valid month
        ("transaction_month", 12, True),  # Valid month
        ("transaction_month", 0, False),  # Invalid month
        ("transaction_month", 13, False),  # Invalid month
        ("transaction_day", 1, True),  # Valid day
        ("transaction_day", 31, True),  # Valid day
        ("transaction_day", 0, False),  # Invalid day
        ("transaction_day", 32, False),  # Invalid day
        ("is_weekend", 0, True),  # Valid weekend flag
        ("is_weekend", 1, True),  # Valid weekend flag
        ("is_weekend", 2, False),  # Invalid weekend flag
        ("avg_transaction_value", 0.01, True),  # Minimum valid
        ("avg_transaction_value", 0, False),  # Zero (gt=0 constraint)
        ("transaction_count", 1, True),  # Minimum valid
        ("transaction_count", 0, False),  # Zero (gt=0 constraint)
        ("Recency", 0, True),  # Minimum valid
        ("Recency", -1, False),  # Negative recency
        ("Frequency", 1, True),  # Minimum valid
        ("Frequency", 0, False),  # Zero frequency
    ],
)
def test_transaction_input_validation(
    client, sample_transaction, mock_model, field, value, should_pass
):
    """Test input validation for various transaction fields."""
    MODEL_STATE["model"] = mock_model
    sample_transaction[field] = value

    response = client.post("/predict", json=sample_transaction)

    if should_pass:
        assert response.status_code in [200, 503]  # 503 if model not loaded
    else:
        assert response.status_code == 422  # Validation error


def test_missing_required_fields(client, mock_model):
    """Test prediction with missing required fields."""
    MODEL_STATE["model"] = mock_model

    incomplete_transaction = {
        "TransactionId": "TXN_001",
        "Amount": 5000.0,
        # Missing many required fields
    }

    response = client.post("/predict", json=incomplete_transaction)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_extra_fields_ignored(client, sample_transaction, mock_model):
    """Test that extra fields in request are ignored."""
    MODEL_STATE["model"] = mock_model

    sample_transaction["extra_field"] = "should_be_ignored"
    sample_transaction["another_extra"] = 12345

    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200  # Should still work


# ============================================================================
# Boundary Value Tests
# ============================================================================


def test_predict_extreme_values(client, sample_transaction, mock_model):
    """Test prediction with extreme but valid values."""
    MODEL_STATE["model"] = mock_model

    # Set extreme values
    sample_transaction["Amount"] = 999999999.99
    sample_transaction["Value"] = 999999999.99
    sample_transaction["total_transaction_value"] = 999999999.99
    sample_transaction["Recency"] = 10000
    sample_transaction["Frequency"] = 100000
    sample_transaction["Monetary"] = 999999999.99

    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200


def test_predict_minimum_values(client, sample_transaction, mock_model):
    """Test prediction with minimum valid values."""
    MODEL_STATE["model"] = mock_model

    # Set minimum values
    sample_transaction["Amount"] = 0.01
    sample_transaction["Value"] = 0.01
    sample_transaction["avg_transaction_value"] = 0.01
    sample_transaction["transaction_count"] = 1
    sample_transaction["Recency"] = 0
    sample_transaction["Frequency"] = 1
    sample_transaction["Monetary"] = 0.01

    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
