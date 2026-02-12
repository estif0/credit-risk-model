"""
Reusable UI components for the Credit Risk Dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


def header_component(title: str, subtitle: str) -> None:
    """
    Render a standard header.

    Args:
        title: Main title
        subtitle: Subtitle/description
    """
    st.markdown(
        f"""
        <div style="padding: 1rem 0; border-bottom: 2px solid #f0f2f6; margin-bottom: 2rem;">
            <h1 style="color: #0e1117; margin-bottom: 0.5rem;">{title}</h1>
            <p style="color: #262730; font-size: 1.2rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None
) -> None:
    """
    Render a metric card.

    Args:
        label: Label of the metric
        value: Value of the metric
        delta: Change in value (optional)
        help_text: Tooltip text (optional)
    """
    st.metric(label=label, value=value, delta=delta, help=help_text)


def feature_importance_chart(feature_importance: pd.DataFrame, top_n: int = 10) -> None:
    """
    Render feature importance bar chart.

    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of features to show
    """
    if feature_importance.empty:
        st.warning("No feature importance data available")
        return

    df = feature_importance.head(top_n).sort_values("importance", ascending=True)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Most Important Features",
        labels={"importance": "Importance Score", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def risk_score_gauge(score: int, min_val: int = 300, max_val: int = 850) -> None:
    """
    Render a gauge chart for credit score.

    Args:
        score: Current credit score
        min_val: Minimum score
        max_val: Maximum score
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Credit Score", "font": {"size": 24}},
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickwidth": 1,
                    "tickcolor": "darkblue",
                },
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [min_val, 550], "color": "red"},
                    {"range": [550, 700], "color": "yellow"},
                    {"range": [700, max_val], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def prediction_result_card(prediction: Dict[str, Any]) -> None:
    """
    Render prediction results prominently.

    Args:
        prediction: Dictionary with prediction details
    """
    risk_prob = prediction.get("risk_probability", 0)
    score = prediction.get("credit_score", 0)
    category = prediction.get("risk_category", "unknown")

    color = "green" if category == "low" else "red"

    st.markdown(
        f"""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f0f2f6; border-left: 5px solid {color};">
            <h3 style="margin-top: 0;">Risk Assessment: <span style="color: {color}">{category.upper()}</span></h3>
            <p><strong>Credit Score:</strong> {score}</p>
            <p><strong>Default Probability:</strong> {risk_prob:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
