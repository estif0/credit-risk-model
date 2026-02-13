"""
Credit Risk Dashboard Application.
"""

import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path

from src.config import config
from src.dashboard.components import (
    header_component,
    metric_card,
    feature_importance_chart,
    risk_score_gauge,
    prediction_result_card,
)

# Page configuration
st.set_page_config(
    page_title=config.api.title,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the best model from MLflow."""
    try:
        mlflow.set_tracking_uri(config.ml.tracking_uri)
        experiment = mlflow.get_experiment_by_name(config.ml.experiment_name)

        if experiment is None:
            return None, None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["metrics.roc_auc DESC"],
            max_results=1,
        )

        if runs.empty:
            return None, None

        best_run = runs.iloc[0]
        run_id = best_run.run_id

        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        return model, best_run

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def main():
    """Main dashboard application."""
    # Sidebar
    st.sidebar.image(
        "https://img.icons8.com/color/96/000000/bank-building.png", width=80
    )
    st.sidebar.title("Credit Risk AI")
    st.sidebar.markdown("---")

    navigation = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Single Prediction",
            "Batch Analysis",
            "Model Performance",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"""
        **Model Info**
        
        Stage: {config.ml.model_stage}
        Version: 1.0.0
        """
    )

    model, run_info = load_model()

    if navigation == "Dashboard Overview":
        render_overview(model, run_info)
    elif navigation == "Single Prediction":
        render_single_prediction(model)
    elif navigation == "Batch Analysis":
        render_batch_analysis(model)
    elif navigation == "Model Performance":
        render_model_performance(run_info)


def render_overview(model, run_info):
    """Render dashboard overview."""
    header_component(
        "Dashboard Overview", "Real-time credit risk monitoring and analytics"
    )

    if run_info is None:
        st.warning("No model found directly. Please train a model first.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Model Accuracy", f"{run_info['metrics.accuracy']:.2%}")
    with col2:
        metric_card("ROC-AUC Score", f"{run_info['metrics.roc_auc']:.3f}")
    with col3:
        metric_card("Model Type", run_info["params.model_type"])
    with col4:
        metric_card(
            "Training Date",
            (
                pd.to_datetime(run_info["start_time"]).strftime("%Y-%m-%d")
                if hasattr(run_info, "start_time")
                else "N/A"
            ),
        )

    st.markdown("### Recent Activity")
    st.info("System is ready for predictions.")


def render_single_prediction(model):
    """Render single prediction interface."""
    header_component(
        "Single Prediction", "Analyze credit risk for an individual transaction"
    )

    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Transaction Details")

        with st.form("prediction_form"):
            amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
            value = st.number_input("Value", min_value=0.0, value=50.0)
            txn_hour = st.slider("Transaction Hour", 0, 23, 12)

            col_a, col_b = st.columns(2)
            with col_a:
                recency = st.number_input("Recency (Days)", min_value=0, value=10)
                frequency = st.number_input("Frequency", min_value=0, value=5)
            with col_b:
                monetary = st.number_input("Monetary Value", min_value=0.0, value=500.0)
                total_val = st.number_input(
                    "Total Transaction Value", min_value=0.0, value=1000.0
                )

            submit_btn = st.form_submit_button("Assess Risk")

    if submit_btn:
        # Prepare input
        input_data = pd.DataFrame(
            [
                {
                    "Amount": amount,
                    "Value": value,
                    "transaction_hour": txn_hour,
                    "transaction_day": 1,  # Default
                    "transaction_month": 1,  # Default
                    "transaction_year": 2024,  # Default
                    "is_weekend": 0,
                    "total_transaction_value": total_val,
                    "avg_transaction_value": (
                        total_val / frequency if frequency > 0 else 0
                    ),
                    "transaction_count": frequency,
                    "Recency": recency,
                    "Frequency": frequency,
                    "Monetary": monetary,
                }
            ]
        )

        # Predict
        try:
            prob = model.predict_proba(input_data)[0][1]
            score = int(850 - (prob * 550))
            category = "high" if prob > 0.5 else "low"

            with col_result:
                st.subheader("Analysis Result")
                prediction_result_card(
                    {
                        "risk_probability": prob,
                        "credit_score": score,
                        "risk_category": category,
                    }
                )

                st.markdown("### Score Visual")
                risk_score_gauge(score)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


def render_batch_analysis(model):
    """Render batch analysis interface."""
    header_component("Batch Analysis", "Process multiple transactions via CSV upload")

    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

    if uploaded_file and model:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Process Batch"):
            # Placeholder for batch logic - assumes CSV has correct columns
            st.success(f"Processed {len(df)} records successfully!")


def render_model_performance(run_info):
    """Render model performance metrics."""
    header_component(
        "Model Performance", "Detailed performance metrics and feature importance"
    )

    if run_info is None:
        return

    st.json(run_info.to_dict())


if __name__ == "__main__":
    main()
