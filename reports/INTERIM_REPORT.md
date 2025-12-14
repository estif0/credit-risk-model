# Interim Report: Credit Risk Probability Model for Alternative Data
## Bati Bank Buy-Now-Pay-Later Service Development

**Project:** Credit Scoring Model Using RFM-Based Behavioral Analytics  
**Author:** Estifanose Sahilu  
**Date:** December 14, 2025  
**Status:** Tasks 1-2 Complete | Feature Engineering In Progress

---

## Executive Summary

This interim report documents my progress on developing a Credit Scoring Model for Bati Bank's buy-now-pay-later partnership with an eCommerce platform. I have successfully completed the business understanding phase and exploratory data analysis (EDA) on 95,662 transactions from the Xente Challenge dataset. My key findings reveal excellent data quality (zero missing values), significant distributional challenges requiring transformation, and clear opportunities for RFM-based customer segmentation. My analysis confirms the dataset's suitability for proxy-based credit risk modeling, positioning me to proceed with feature engineering and model development.

---

## 1. Understanding and Defining the Business Objective

### 1.1 Business Context and Strategic Importance

Bati Bank is launching a buy-now-pay-later service in partnership with an established eCommerce company, requiring a sophisticated credit scoring model to assess customer creditworthiness. Unlike traditional lending scenarios with historical default data, I face a **cold start problem**: I must build a predictive model using only transactional behavioral data, without direct labels indicating which customers have defaulted on loans.

This project addresses a critical business need: **enable responsible lending decisions** that balance revenue growth (approving creditworthy customers) with risk management (avoiding defaults that could threaten regulatory capital requirements).

### 1.2 Basel II Capital Accord and Regulatory Compliance

The **Basel II Capital Accord** establishes international standards for banking supervision, with three pillars:
1. **Minimum Capital Requirements** based on credit risk, market risk, and operational risk
2. **Supervisory Review Process** ensuring adequate capital and risk management
3. **Market Discipline** through disclosure requirements

**Impact on My Model Design:**

Basel II's emphasis on **quantitative risk measurement** directly influences my development approach in three critical ways:

1. **Model Transparency Requirements:** I must demonstrate clear understanding of how my model assigns risk. This necessitates:
   - Comprehensive documentation of methodology and assumptions
   - Explainable feature contributions to risk scores
   - Clear rationale for proxy variable selection
   - Traceable decision boundaries

2. **Validation and Back-Testing:** My model must undergo rigorous validation:
   - Performance metrics across different customer segments
   - Regular monitoring for model drift
   - Comparison against benchmark models
   - Documentation of all assumptions and limitations

3. **Risk Quantification Standards:** I must provide:
   - Probability of Default (PD) estimates for regulatory capital calculations
   - Confidence intervals for predictions
   - Calibration evidence across the risk spectrum

**Strategic Decision:** These requirements favor **interpretable models** (Logistic Regression with WoE) as my baseline, with more complex models (Gradient Boosting) requiring extensive documentation and post-hoc explanation techniques (SHAP, LIME) for regulatory acceptance.

### 1.3 Proxy Variable Necessity and Business Risks

**Why Proxy Variables Are Necessary:**

Traditional credit scoring relies on historical default data—records of customers who failed to repay loans. My situation presents three challenges:

1. **No Historical Default Data:** As a new service, I have zero observations of loan defaults
2. **Only Transactional Data Available:** I possess eCommerce behavioral data (purchase patterns, amounts, timing, channels)
3. **Time Pressure:** I cannot delay service launch 12-24 months to accumulate default history

**My Solution: RFM-Based Proxy Target**

I create a proxy for credit risk through **RFM (Recency, Frequency, Monetary) analysis**:
- **Recency:** Days since last transaction (disengaged customers may indicate higher risk)
- **Frequency:** Number of transactions (active customers demonstrate commitment)
- **Monetary:** Total transaction value (indicates financial capacity)

**Hypothesis:** Customers with high recency (long time since last purchase), low frequency (few transactions), and low monetary value (small purchases) represent higher credit risk due to low engagement and limited demonstrated financial capacity.

**Critical Business Risks:**

1. **Unvalidated Assumption Risk:**
   - I assume engagement correlates with creditworthiness—an untested hypothesis
   - Low shopping frequency ≠ necessarily indicates inability to repay
   - Could systematically disadvantage certain customer segments (e.g., infrequent high-value purchasers)

2. **Regulatory Scrutiny:**
   - Basel II requires models based on "sound statistical evidence"
   - My proxy approach lacks ground truth validation
   - May face challenges in regulatory approval processes

3. **Discrimination and Bias Risk:**
   - May unfairly penalize new customers or those with changing patterns
   - Could create unintended bias against specific demographic segments
   - Risk of excluding creditworthy individuals based on shopping behavior

4. **Model Drift Inevitability:**
   - As I collect actual default data, engagement-default relationship may prove weaker than assumed
   - Will require complete model retraining within 12-24 months
   - Initial predictions have uncertain real-world validity

**Mitigation Strategy:** Implement rigorous monitoring from day one, collect actual default/repayment data for all loans, maintain human review for borderline cases, and plan for model replacement once sufficient ground truth data exists (typically 12-24 months).

### 1.4 Model Trade-offs: Interpretability vs. Performance

**Logistic Regression with Weight of Evidence (WoE)**

*Advantages:*
- **Regulatory Fit:** Excellent alignment with Basel II transparency requirements
- **Interpretability:** Each coefficient clearly shows directional impact on risk
- **WoE Benefits:** Monotonic feature-target relationships, business-understandable binning, Information Value (IV) for feature selection
- **Operational:** Fast inference, easy debugging, simple stakeholder communication

*Disadvantages:*
- **Linear Limitations:** Cannot capture complex non-linear relationships or feature interactions without manual engineering
- **Lower Predictive Power:** May miss subtle risk patterns
- **Manual Effort:** Requires extensive feature engineering

**Gradient Boosting (XGBoost, LightGBM)**

*Advantages:*
- **Predictive Performance:** Superior ability to capture non-linear relationships and feature interactions
- **Automatic Learning:** Discovers optimal feature combinations without manual specification
- **Technical Robustness:** Built-in missing value handling, outlier resistance

*Disadvantages:*
- **Regulatory Challenge:** Black-box nature conflicts with Basel II transparency requirements
- **Complexity:** Difficult to explain individual predictions to regulators or customers
- **Maintenance:** Harder to debug and update as business rules evolve
- **Overfitting Risk:** Can memorize training data, especially with limited samples

**My Hybrid Approach:**

Given my context (proxy-based target, regulatory requirements, new service), I will:

1. **Baseline:** Start with Logistic Regression + WoE to establish an interpretable foundation and build regulatory trust
2. **Parallel Development:** Build Gradient Boosting models with SHAP explanations for comparison
3. **Governance:** If deploying complex models, implement extensive documentation, human-in-the-loop review for borderline cases, and maintain Logistic Regression as validation
4. **Evolution:** Reassess model choice once actual default data is available

**Conclusion:** In a regulated environment with proxy targets, I prioritize **interpretability and validation** over marginal performance gains.

---

## 2. Exploratory Data Analysis: Completed Work and Key Findings

### 2.1 Dataset Overview

**Data Source:** Kaggle Xente Challenge - eCommerce transaction data  
**Dimensions:** 95,662 transactions × 16 features  
**Time Period:** November 15, 2018 - February 13, 2019 (90 days)  
**Data Quality:** Exceptional - zero missing values across all features

**Feature Categories:**
- **Numerical Features (5):** CountryCode, Amount, Value, PricingStrategy, FraudResult
- **Categorical Features (11):** TransactionId, BatchId, AccountId, SubscriptionId, CustomerId, CurrencyCode, ProviderId, ProductId, ProductCategory, ChannelId, TransactionStartTime

### 2.2 Summary Statistics and Distribution Analysis

**Numerical Features - Key Statistics:**

| Feature         | Mean     | Median   | Std Dev    | Skewness | Interpretation              |
| --------------- | -------- | -------- | ---------- | -------- | --------------------------- |
| Amount          | 6,717.85 | 1,000.00 | 123,306.80 | 51.098   | Extreme right-skewed        |
| Value           | 9,900.58 | 1,000.00 | 123,122.09 | 51.290   | Extreme right-skewed        |
| PricingStrategy | 2.26     | 2.00     | 0.73       | 1.659    | Moderate right-skewed       |
| FraudResult     | 0.00     | 0.00     | 0.04       | 22.196   | Highly skewed (rare events) |

**Critical Observations:**
- Transaction amounts show severe right skewness (skewness > 51), indicating most transactions are small with a long tail of large transactions
- Mean >> Median confirms the presence of extreme outliers
- High coefficient of variation suggests diverse customer segments

![Figure 1: Distribution Analysis of Amount and Value](./figures/distribution_analysis.png)
*Figure 1: Histogram and log-transformed distributions of Amount and Value features showing extreme right-skewness (51+). The log transformation reveals a more normal distribution, essential for linear model performance.*

**Categorical Features - Cardinality Analysis:**

| Feature         | Unique Values | Cardinality Type | Encoding Strategy     |
| --------------- | ------------- | ---------------- | --------------------- |
| ProductCategory | 6             | Low              | One-Hot Encoding      |
| ChannelId       | 5             | Low              | One-Hot Encoding      |
| CurrencyCode    | 1             | Constant         | Drop                  |
| ProviderId      | 144           | High             | Label/Target Encoding |
| ProductId       | 97            | High             | Label/Target Encoding |
| CustomerId      | 6,393         | Very High        | Aggregation Key (RFM) |

### 2.3 Correlation Analysis

**Key Finding:** Amount and Value show near-perfect correlation (r = 0.990)

This represents multicollinearity that must be addressed. Value is the absolute value of Amount (Amount can be negative for credits, Value is always positive). Using both features would cause coefficient instability in linear models.

**Additional Correlations:**
- Amount/Value with FraudResult: Moderate positive correlation (r ≈ 0.56), suggesting fraud detection may be related to transaction size
- Weak correlations elsewhere indicate features provide largely independent information

![Figure 2: Correlation Heatmap](./figures/correlation_heatmap.png)
*Figure 2: Correlation matrix for numerical features. The near-perfect correlation (0.990) between Amount and Value necessitates removing one feature to prevent multicollinearity in linear models.*

### 2.4 Temporal Patterns

**Transaction Timing Analysis:**

- **Peak Hours:** 4 PM - 5 PM (16:00-17:00) show highest transaction volume
- **Day of Week:** Friday shows significantly higher activity (nearly 2× other days)
- **Monthly Pattern:** December (Month 12) and November (Month 11) show elevated volumes, likely holiday shopping
- **Business Hours Dominance:** 90% of transactions occur between 6 AM and 11 PM

**Implications:** Temporal features (hour, day, month) will be valuable for modeling, particularly for fraud detection and customer behavior profiling.

![Figure 3: Temporal Transaction Patterns](./figures/temporal_analysis.png)
*Figure 3: Transaction patterns by hour, day of week, month, and year. Clear peak hours (4-5 PM), Friday dominance (2× other days), and holiday seasonality (Nov-Dec) provide strong behavioral signals for credit risk assessment.*

### 2.5 Missing Values and Outlier Analysis

**Missing Values:** Zero missing values across all 95,662 transactions—exceptional data quality eliminating need for imputation strategies.

**Outlier Detection (IQR Method):**

| Feature         | Total Outliers | Outlier % | Recommendation                      |
| --------------- | -------------- | --------- | ----------------------------------- |
| Amount          | 14,349         | 15.0%     | Log transformation                  |
| Value           | 14,349         | 15.0%     | Log transformation                  |
| PricingStrategy | 7,165          | 7.5%      | Winsorization at 95th percentile    |
| FraudResult     | 191            | 0.2%      | Keep as-is (legitimate rare events) |

**Critical Note:** In credit risk contexts, outliers often represent legitimate extreme behaviors (high spenders, unusual patterns) that may be highly predictive. I will use transformation rather than removal.

![Figure 4: Box Plots for Outlier Detection](./figures/outlier_boxplots.png)
*Figure 4: Box plots showing outlier distribution across numerical features. Amount and Value show 15% outliers, while FraudResult shows 0.2% rare events. Transformation strategy preferred over removal to preserve predictive signals.*

### 2.6 Top 5 Key Insights

**Insight #1: Perfect Data Quality Enables Direct Feature Engineering**
- Zero missing values across all features indicates robust data collection
- Eliminates imputation complexity and associated bias risks
- Allows immediate focus on feature engineering and model development

**Insight #2: Severe Distributional Skewness Requires Transformation**
- Amount and Value show extreme right-skewness (51+), with long tails of large transactions
- Linear models will fail without log transformation
- Suggests diverse customer segments: small frequent purchasers vs. large infrequent buyers
- Requires robust scaling (StandardScaler) over MinMaxScaler

**Insight #3: Multicollinearity Between Amount and Value Must Be Resolved**
- Near-perfect correlation (0.990) creates redundancy
- Solution: Use Value for magnitude, create binary `is_credit` feature for direction
- Drop Amount from final feature set to avoid coefficient instability

**Insight #4: High Cardinality Features Require Sophisticated Encoding**
- ProviderId (144 unique), ProductId (97 unique) cannot use one-hot encoding
- CustomerId (6,393 unique) is an aggregation key for RFM, not a direct feature
- Require Label/Target Encoding or WoE transformation for predictive power

**Insight #5: Dataset Ideal for RFM-Based Proxy Target Creation**
- Contains all necessary components: CustomerId, TransactionStartTime, Value, transaction counts
- 90-day observation period provides sufficient data for recency calculations
- Clear temporal patterns enable rich feature engineering beyond basic RFM

![Figure 5: Categorical Feature Distribution](./figures/categorical_distributions.png)
*Figure 5: Distribution of categorical features showing cardinality levels. Low-cardinality features (ProductCategory: 6, ChannelId: 5) suitable for one-hot encoding, while high-cardinality features (ProviderId: 144, ProductId: 97) require target encoding or WoE transformation.*

---

## 3. Next Steps and Roadmap for Remaining Tasks

### 3.1 Task 3: Feature Engineering (In Progress)

**Objective:** Transform raw transaction data into predictive features for modeling.

**My Approach:**

**Aggregate Features (Per Customer):**
- Transaction count: Number of transactions per CustomerId
- Total transaction value: Sum of Value per customer
- Average transaction value: Mean of Value per customer
- Transaction value standard deviation: Volatility indicator
- Transaction value range: Max - Min per customer

**Temporal Feature Extraction:**
- Hour of day (0-23): Peak activity indicators
- Day of week (0-6): Weekday vs weekend behavior
- Month (1-12): Seasonal patterns
- Days since first transaction: Customer tenure
- Days between transactions: Transaction frequency proxy

**Categorical Encoding Strategy:**
- **One-Hot Encoding:** ChannelId, ProductCategory (low cardinality)
- **Label Encoding:** ProviderId, ProductId (preserve ordinal relationships if any)
- **Target Encoding:** High-cardinality features encoded by mean target value
- **WoE Transformation:** Create bins with monotonic relationship to target

**Feature Scaling:**
- StandardScaler for tree-based models (handles outliers)
- Log transformation for Amount/Value before scaling
- Create `is_credit` binary feature (1 if Amount < 0, else 0)

**Implementation:** I'll use sklearn.pipeline.Pipeline to ensure reproducibility and prevent data leakage.

### 3.2 Task 4: RFM Analysis and Proxy Target Variable Creation

**My RFM Metric Calculation:**
```
For each CustomerId:
  Recency = (Reference_Date - Last_Transaction_Date).days
  Frequency = Count(Transactions)
  Monetary = Sum(Value) OR Mean(Value)
```

**Clustering Approach:**
1. Scale RFM features using StandardScaler
2. Apply K-Means clustering (k=3) to segment customers
3. Profile each cluster by RFM characteristics
4. Identify "high-risk" cluster: high recency + low frequency + low monetary
5. Assign binary target: `is_high_risk` = 1 for risky cluster, 0 otherwise

**Validation:**
- Visualize clusters in 3D space (Recency, Frequency, Monetary)
- Document cluster sizes and characteristics
- Justify high-risk label assignment with business logic

**My Integration Plan:**
- Merge `is_high_risk` back to transaction-level dataset on CustomerId
- Save processed dataset with target variable to `data/processed/`

### 3.3 Task 5: Model Training and Experiment Tracking

**My Model Implementation Plan:**
1. **Logistic Regression + WoE:** Interpretable baseline, regulatory-friendly
2. **Decision Tree:** Interpretable non-linear model with clear decision rules
3. **Random Forest:** Ensemble method handling interactions without overfitting
4. **Gradient Boosting (XGBoost):** High-performance model with SHAP explanations

**Hyperparameter Tuning:**
- GridSearchCV or RandomizedSearchCV for systematic search
- Cross-validation (5-fold stratified) to prevent overfitting
- Focus on precision-recall trade-off given business context

**MLflow Integration:**
```python
mlflow.set_experiment("credit-risk-modeling")
with mlflow.start_run(run_name="logistic_regression_woe"):
    mlflow.log_params({...})  # Log hyperparameters
    mlflow.log_metrics({...})  # Log accuracy, precision, recall, F1, ROC-AUC
    mlflow.sklearn.log_model(model, "model")  # Log trained model
```

**Evaluation Metrics:**
- **Accuracy:** Overall correctness
- **Precision:** Minimize false positives (rejecting good customers)
- **Recall:** Minimize false negatives (approving bad customers)
- **F1 Score:** Balance between precision and recall
- **ROC-AUC:** Discrimination ability across thresholds

**Model Selection Criteria:**
- Interpretability (weighted heavily for regulatory compliance)
- Predictive performance on hold-out test set
- Stability across cross-validation folds
- Business alignment (cost of false positives vs false negatives)

### 3.4 Task 6: Model Deployment and CI/CD

**FastAPI Development:**
```python
@app.post("/predict", response_model=RiskResponse)
async def predict_risk(transaction: TransactionRequest):
    # Feature engineering pipeline
    # Model prediction
    # Return: risk_probability, risk_category, credit_score
```

**Docker Containerization:**
- Multi-stage Dockerfile for optimized image size
- Include MLflow model registry integration
- Environment variable configuration for flexibility

**CI/CD Pipeline (GitHub Actions):**
1. **Linting:** flake8, black for code quality
2. **Testing:** pytest with minimum 80% coverage
3. **Build:** Docker image creation
4. **Deploy:** Automated deployment on merge to main

**Deliverables Timeline:**
- Feature Engineering: December 14 (Today)
- RFM & Target Creation: December 14-15
- Model Training: December 15-16
- API & Deployment: December 16 (Final deadline)

---

## 4. Conclusion

I have successfully completed the foundational phases of this credit risk modeling project, establishing both business understanding and technical groundwork. My EDA confirms the dataset's exceptional quality and suitability for RFM-based proxy target creation, while also identifying critical challenges (distributional skewness, multicollinearity) that inform my feature engineering strategy.

**My Key Achievements:**
- Comprehensive business context aligned with Basel II requirements
- Complete EDA with zero missing values and clear insights
- Validated approach for proxy target variable creation
- Defined roadmap for remaining implementation tasks

**My Risk Awareness:**
I acknowledge the limitations of proxy-based modeling and commit to rigorous monitoring, transparent documentation, and planned model evolution as actual default data becomes available.

**My Next 48 Hours:**
My immediate focus is feature engineering and RFM analysis, positioning me to deliver a complete, deployable credit scoring model by the December 16 deadline while maintaining the interpretability and documentation standards required for regulatory compliance.

---

**Report prepared by:** Estifanose Sahilu  
**Document Version:** 1.0  
**Submission Date:** December 14, 2025, 8:00 PM UTC  
**Project Repository:** [Credit Risk Model GitHub Repository]
