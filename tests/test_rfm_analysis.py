"""
Unit tests for rfm_analysis module.

Tests the RFMAnalyzer class and RFM calculation functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.rfm_analysis import RFMAnalyzer, run_rfm_pipeline


@pytest.fixture
def sample_rfm_data():
    """Fixture providing sample RFM transaction data."""
    np.random.seed(42)

    customers = ["CUST_001", "CUST_002", "CUST_003", "CUST_004", "CUST_005"]
    dates = []
    customer_ids = []
    values = []

    # Create different patterns for different customers
    for i, cust in enumerate(customers):
        # Different transaction patterns
        n_transactions = np.random.randint(5, 20)
        start_date = datetime(2024, 1, 1) - timedelta(days=np.random.randint(1, 90))

        for _ in range(n_transactions):
            dates.append(start_date + timedelta(days=np.random.randint(0, 60)))
            customer_ids.append(cust)
            values.append(np.random.exponential(scale=1000))

    return pd.DataFrame(
        {
            "TransactionId": [f"TXN_{i:04d}" for i in range(len(dates))],
            "CustomerId": customer_ids,
            "TransactionStartTime": dates,
            "Value": values,
        }
    )


def test_rfm_analyzer_initialization():
    """Test RFMAnalyzer initialization."""
    analyzer = RFMAnalyzer(n_clusters=3, random_state=42)

    assert analyzer.n_clusters == 3
    assert analyzer.random_state == 42
    assert analyzer.reference_date is None
    assert analyzer.kmeans is None


def test_calculate_rfm_metrics(sample_rfm_data):
    """Test RFM metric calculation."""
    analyzer = RFMAnalyzer()
    rfm_df = analyzer.calculate_rfm_metrics(sample_rfm_data)

    # Check required columns exist
    assert "CustomerId" in rfm_df.columns
    assert "Recency" in rfm_df.columns
    assert "Frequency" in rfm_df.columns
    assert "Monetary" in rfm_df.columns

    # Check one row per customer
    assert len(rfm_df) == sample_rfm_data["CustomerId"].nunique()

    # Validate metric ranges
    assert (rfm_df["Recency"] >= 0).all()
    assert (rfm_df["Frequency"] > 0).all()
    assert (rfm_df["Monetary"] > 0).all()


def test_calculate_rfm_metrics_missing_column():
    """Test RFM calculation raises error for missing columns."""
    analyzer = RFMAnalyzer()
    df = pd.DataFrame({"CustomerId": ["A", "B"]})

    with pytest.raises(KeyError):
        analyzer.calculate_rfm_metrics(df)


def test_calculate_rfm_metrics_empty_dataframe():
    """Test RFM calculation raises error for empty DataFrame."""
    analyzer = RFMAnalyzer()
    df = pd.DataFrame(columns=["CustomerId", "TransactionStartTime", "Value"])

    with pytest.raises(ValueError):
        analyzer.calculate_rfm_metrics(df)


def test_perform_clustering(sample_rfm_data):
    """Test K-Means clustering on RFM metrics."""
    analyzer = RFMAnalyzer(n_clusters=3)
    rfm_df = analyzer.calculate_rfm_metrics(sample_rfm_data)

    result = analyzer.perform_clustering(rfm_df)

    # Check cluster column exists
    assert "Cluster" in result.columns

    # Check cluster values are valid
    assert result["Cluster"].nunique() <= 3
    assert result["Cluster"].min() >= 0
    assert result["Cluster"].max() < 3

    # Check analyzer state
    assert analyzer.kmeans is not None
    assert analyzer.cluster_profiles is not None
    assert analyzer.high_risk_cluster is not None


def test_perform_clustering_with_nans():
    """Test clustering raises error with NaN values."""
    analyzer = RFMAnalyzer()
    df = pd.DataFrame(
        {
            "CustomerId": ["A", "B", "C"],
            "Recency": [10, 20, np.nan],
            "Frequency": [5, 10, 15],
            "Monetary": [1000, 2000, 3000],
        }
    )

    with pytest.raises(ValueError):
        analyzer.perform_clustering(df)


def test_assign_proxy_target(sample_rfm_data):
    """Test proxy target assignment."""
    analyzer = RFMAnalyzer(n_clusters=3)
    rfm_df = analyzer.calculate_rfm_metrics(sample_rfm_data)
    rfm_df = analyzer.perform_clustering(rfm_df)

    result = analyzer.assign_proxy_target(rfm_df)

    # Check target column exists
    assert "is_high_risk" in result.columns

    # Check binary values
    assert result["is_high_risk"].isin([0, 1]).all()

    # Check at least one customer in each class
    assert result["is_high_risk"].nunique() == 2


def test_assign_proxy_target_without_clustering():
    """Test proxy target assignment fails without clustering."""
    analyzer = RFMAnalyzer()
    df = pd.DataFrame(
        {
            "CustomerId": ["A", "B"],
            "Recency": [10, 20],
            "Frequency": [5, 10],
            "Monetary": [1000, 2000],
        }
    )

    with pytest.raises(ValueError):
        analyzer.assign_proxy_target(df)


def test_get_cluster_summary(sample_rfm_data):
    """Test cluster summary generation."""
    analyzer = RFMAnalyzer(n_clusters=3)
    rfm_df = analyzer.calculate_rfm_metrics(sample_rfm_data)
    rfm_df = analyzer.perform_clustering(rfm_df)

    summary = analyzer.get_cluster_summary()

    # Check summary structure
    assert "n_clusters" in summary
    assert "high_risk_cluster" in summary
    assert "cluster_profiles" in summary
    assert "reference_date" in summary

    assert summary["n_clusters"] == 3
    assert isinstance(summary["high_risk_cluster"], int)


def test_run_rfm_pipeline(sample_rfm_data, tmp_path):
    """Test complete RFM pipeline execution."""
    output_file = tmp_path / "rfm_output.csv"

    rfm_results, analyzer = run_rfm_pipeline(
        sample_rfm_data, n_clusters=3, save_results=True, output_path=str(output_file)
    )

    # Check results DataFrame
    assert "Recency" in rfm_results.columns
    assert "Frequency" in rfm_results.columns
    assert "Monetary" in rfm_results.columns
    assert "Cluster" in rfm_results.columns
    assert "is_high_risk" in rfm_results.columns

    # Check analyzer state
    assert analyzer.kmeans is not None
    assert analyzer.high_risk_cluster is not None

    # Check file was created
    assert output_file.exists()


def test_clustering_deterministic():
    """Test that clustering produces deterministic results with same random_state."""
    df = pd.DataFrame(
        {
            "CustomerId": [f"CUST_{i:03d}" for i in range(100)],
            "TransactionStartTime": pd.date_range("2024-01-01", periods=100),
            "Value": np.random.exponential(1000, 100),
        }
    )

    # Run twice with same random state
    analyzer1 = RFMAnalyzer(n_clusters=3, random_state=42)
    rfm1 = analyzer1.calculate_rfm_metrics(df)
    rfm1 = analyzer1.perform_clustering(rfm1)

    analyzer2 = RFMAnalyzer(n_clusters=3, random_state=42)
    rfm2 = analyzer2.calculate_rfm_metrics(df)
    rfm2 = analyzer2.perform_clustering(rfm2)

    # Results should be identical
    pd.testing.assert_series_equal(rfm1["Cluster"], rfm2["Cluster"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
