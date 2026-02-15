# Credit Risk AI - Business User Guide

## Overview
The Credit Risk AI system provides real-time credit scoring for buy-now-pay-later transactions. This guide explains how to use the interactive dashboard to make data-driven credit decisions.

## Accessing the Dashboard
1. Open your web browser and navigate to the dashboard URL (typically `http://localhost:8501`).
2. You will see the **Dashboard Overview** screen showing system status and model performance metrics.

## Features

### 1. Single Customer Prediction
Use this feature to assess credit risk for an individual customer transaction.

**Steps:**
1. Select **Single Prediction** from the sidebar navigation.
2. Enter transaction details:
   - **Amount**: The transaction value.
   - **Recency**: Days since last transaction.
   - **Frequency**: Number of past transactions.
   - **Monetary**: Total value of past transactions.
3. Click **Assess Risk**.

**Interpreting Results:**
- **Risk Category**: 
  - <span style="color:green">**LOW**</span>: Safe to approve credit.
  - <span style="color:red">**HIGH**</span>: High risk of default; manual review recommended.
- **Credit Score**: A score between 300-850. Higher is better.
- **Probability**: The calculated likelihood of default (0-100%).

### 2. Batch Analysis
Process multiple transactions at once.

**Steps:**
1. Select **Batch Analysis** from the sidebar.
2. Upload a CSV file containing transaction data.
3. Click **Process Batch** to generate scores for all records.
4. Download the results CSV.

### 3. Model Performance
View technical metrics to understand model reliability.

- **ROC-AUC**: Measure of the model's ability to distinguish between high and low risk. (Target: > 0.8)
- **Accuracy**: Overall correctness of predictions.

## Troubleshooting
- **"Model not loaded"**: Ensure the backend API is running and a model has been trained.
- **"Prediction failed"**: Check if all input fields contain valid numerical values.

## Support
For technical assistance, contact the Data Science Team at `support@batibank.com`.
