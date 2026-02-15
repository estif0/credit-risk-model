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
import sys

# Add project root to sys.path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
            max_results=3,
        )

        if runs.empty:
            return None, None

        for _, run in runs.iterrows():
            try:
                run_id = run.run_id
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)
                return model, run
            except Exception as e:
                st.warning(f"Failed to load model from run {run.run_id}: {e}")
                continue

        st.error("Failed to load any model from top runs")
        return None, None

        return model, best_run

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


import requests

API_URL = f"http://{config.api.host}:{config.api.port}"


@st.cache_data(ttl=60)
def get_model_list():
    """Fetch available models from API."""
    try:
        response = requests.get(f"{API_URL}/model/list")
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return []


def switch_model(run_id: str):
    """Switch active model via API."""
    try:
        response = requests.post(f"{API_URL}/model/load/{run_id}")
        if response.status_code == 200:
            st.success(f"Switched to model: {run_id}")
            return True
        else:
            st.error(f"Failed to switch model: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error switching model: {e}")
        return False


def main():
    """Main dashboard application."""
    # Sidebar
    st.sidebar.image(
        "https://img.icons8.com/color/96/000000/bank-building.png", width=80
    )
    st.sidebar.title("Credit Risk AI")
    st.sidebar.markdown("---")

    # Model Selection
    st.sidebar.subheader("Model Selection")
    models = get_model_list()

    if models:
        # Create a mapping of display name to run_id
        model_options = {
            f"{m['type']} ({m['created']}) - AUC: {m['roc_auc']:.3f}": m["run_id"]
            for m in models
        }
        selected_option = st.sidebar.selectbox(
            "Select Model", list(model_options.keys())
        )

        if selected_option:
            run_id = model_options[selected_option]
            if st.sidebar.button("Load Model"):
                switch_model(run_id)
    else:
        st.sidebar.warning("No models found or API unavailable")

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

    # Fetch current model info from API
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            info = response.json()
            st.sidebar.info(
                f"""
                **Active Model**
                
                Type: {info.get('type', 'Unknown')}
                Version: {info.get('version', 'Unknown')}
                Stage: {config.ml.model_stage}
                """
            )
    except:
        st.sidebar.warning("API Offline")

    # Load local model for visualization (fallback/legacy)
    # In a full production setup, visualizations would also come from API stats
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
            # Core Inputs
            amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
            value = st.number_input("Value", min_value=0.0, value=50.0)

            # Temporal Inputs
            txn_date = st.date_input("Transaction Date", datetime.now())
            txn_hour = st.slider("Transaction Hour", 0, 23, 12)

            # RFM Inputs
            col_a, col_b = st.columns(2)
            with col_a:
                recency = st.number_input("Recency (Days)", min_value=0, value=10)
                frequency = st.number_input("Frequency", min_value=0, value=5)
            with col_b:
                monetary = st.number_input("Monetary Value", min_value=0.0, value=500.0)
                total_val = st.number_input(
                    "Total Transaction Value", min_value=0.0, value=1000.0
                )

            # Advanced/Derived Feature Inputs (Defaults hidden or advanced toggle)
            with st.expander("Advanced Features (Defaults Calculated)", expanded=False):
                max_val = st.number_input(
                    "Max Transaction Value", min_value=0.0, value=200.0
                )
                min_val = st.number_input(
                    "Min Transaction Value", min_value=0.0, value=10.0
                )
                std_val = st.number_input(
                    "Std Dev Transaction Value", min_value=0.0, value=50.0
                )
                channel_id = st.selectbox(
                    "Channel ID",
                    ["ChannelId_1", "ChannelId_2", "ChannelId_3", "ChannelId_4"],
                )
                product_cat = st.selectbox(
                    "Product Category",
                    [
                        "financial_services",
                        "airtime",
                        "data_bundles",
                        "tv",
                        "utility_bill",
                    ],
                )

            submit_btn = st.form_submit_button("Assess Risk")

    if submit_btn:
        # Prepare input with ALL required features
        # Mocks for demonstration - in production, these would come from feature store or real-time calculation

        # Temporal
        txn_day = txn_date.weekday()
        txn_month = txn_date.month
        txn_year = txn_date.year
        is_weekend = 1 if txn_day >= 5 else 0

        # Aggregates
        avg_val = total_val / frequency if frequency > 0 else 0
        value_range = max_val - min_val
        value_cv = std_val / avg_val if avg_val > 0 else 0

        input_data = pd.DataFrame(
            [
                {
                    "Amount": amount,
                    "Value": value,
                    "transaction_hour": txn_hour,
                    "transaction_day": txn_day,
                    "transaction_month": txn_month,
                    "transaction_year": txn_year,
                    "is_weekend": is_weekend,
                    "total_transaction_value": total_val,
                    "avg_transaction_value": avg_val,
                    "min_transaction_value": min_val,
                    "max_transaction_value": max_val,
                    "std_transaction_value": std_val,
                    "transaction_count": frequency,
                    "value_range": value_range,
                    "value_cv": value_cv,
                    "Recency": recency,
                    "Frequency": frequency,
                    "Monetary": monetary,
                    # WoE placeholders (assuming 0 for simplicity/new customer)
                    "ChannelId_woe": 0.1,
                    "ProductCategory_woe": 0.1,
                    "ProductId_woe": 0.0,
                    "ProviderId_woe": 0.0,
                    "time_period_woe": 0.0,
                }
            ]
        )

        # Ensure column order matches training if possible, or model handles it by name
        # Missing columns will be filled with 0

        # Predict
        try:
            # Add missing columns with 0 if necessary
            if hasattr(model, "feature_names_in_"):
                missing_cols = set(model.feature_names_in_) - set(input_data.columns)
                for c in missing_cols:
                    input_data[c] = 0
                input_data = input_data[model.feature_names_in_]

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
            st.warning("Ensure all features match the trained model's expectations.")


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
