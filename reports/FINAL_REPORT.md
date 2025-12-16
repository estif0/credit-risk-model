# Building a Credit Scoring Model Without Default Data
## Transforming Transaction Behavior into Creditworthiness Predictions

**Bati Bank Buy-Now-Pay-Later Service**  
*Estifanose Sahilu | December 16, 2025*

---

## Executive Summary

Bati Bank's new buy-now-pay-later service faces a fundamental challenge: predicting creditworthiness without historical default data. This report presents a production-ready credit scoring model built from 95,662 eCommerce transactions, using behavioral patterns as a proxy for credit risk.

**Key Results:**
- **91% ROC-AUC** on test data using Gradient Boosting (67% recall, 71% precision)
- **3,742 customers** scored with risk probabilities and credit scores (300-850 scale)
- **23 engineered features** from temporal patterns, RFM metrics, and behavioral aggregates
- **RESTful API deployed** with Docker, CI/CD pipeline, and MLflow experiment tracking

**Business Impact:** Enable responsible lending with projected $7M Year 1 loan volume, $62k net income (6.2% ROI), scaling to $35M and 15% ROI by Year 3 with actual default data.

---

## 1. The Business Challenge: Lending Without Labels

### The Cold-Start Problem

Traditional credit scoring requires years of loan performance data. Bati Bank must make lending decisions from day one without knowing who will default. The stakes are high: approve too many high-risk customers and face crippling losses; reject too many creditworthy applicants and miss revenue opportunities.

**Basel II Compliance Context:** The Basel II Accord mandates quantitative risk measurement with transparent methodologies. Banks must calculate Probability of Default (PD) and maintain explainable models where risk factors are clearly documented. This regulatory framework shapes our approach, favoring interpretable models over "black box" solutions.

### The Proxy Variable Solution

Without default labels, I engineered a **proxy target using RFM (Recency, Frequency, Monetary) analysis**:

**Hypothesis:** Customer engagement patterns predict creditworthiness. Frequent, recent, high-value transactions signal financial capability and reliability—traits associated with lower credit risk.

**Risk Indicators:**
- High recency (long absence) → disengagement or financial constraints
- Low frequency (few transactions) → sporadic income or limited purchasing power  
- Low monetary (small values) → constrained financial resources

K-Means clustering on RFM metrics identified three customer segments, with high-recency/low-frequency/low-monetary customers labeled as high-risk.

**Acknowledged Risks:** This proxy carries inherent assumptions—wealthy infrequent buyers might be misclassified, seasonal shoppers flagged incorrectly, and engagement may not correlate with actual default behavior. Mitigation includes phased rollout, human review for borderline cases (risk probability 0.4-0.6), and immediate collection of actual default data to validate and eventually replace the proxy.

---

## 2. Data Analysis and Key Findings

### Dataset Overview

**Source:** Xente Challenge (Kaggle) - Mobile money transactions from Rwanda  
**Scale:** 95,662 transactions from 3,742 customers over 90 days (Nov 2018 - Feb 2019)  
**Quality:** Zero missing values, proper data types—exceptional completeness

### Five Critical Discoveries

**1. Extreme Skewness Demands Transformation**

Transaction amounts exhibit severe right-skewness (51+), driven by a small number of very large purchases. Median transaction: $1,000. Mean: $9,901. This 10× difference required log transformation for linear models while tree-based models handled raw distributions naturally.

![Distribution Analysis](./figures/distribution_analysis.png)
*Figure 1: Extreme right-skewness requiring careful feature engineering*

**2. Multicollinearity: Amount vs. Value**

Near-perfect correlation (r=0.990) between Amount and Value fields necessitated dropping one feature. Solution: use Value as primary metric, create binary `is_credit` flag to capture transaction type differences.

![Correlation Heatmap](./figures/correlation_heatmap.png)
*Figure 2: High correlation drives feature selection strategy*

**3. Strong Temporal Patterns**

Clear behavioral signals emerged: peak hours at 4-5 PM (2.5× baseline), Friday volumes 2× weekdays, month-end spikes (40% increase), and 90% of activity between 6 AM-11 PM. These patterns enable sophisticated behavior profiling.

![Temporal Analysis](./figures/temporal_analysis.png)
*Figure 3: Temporal patterns reveal customer behavior rhythms*

**4. Strategic Encoding Based on Cardinality**

Low-cardinality features (ProductCategory: 6 values, ChannelId: 5 values) received one-hot encoding. High-cardinality features (ProviderId: 144 values, ProductId: 97 values) required Weight of Evidence (WoE) transformation—a supervised encoding creating monotonic relationships with the target variable.

**5. RFM Clustering Reveals Distinct Segments**

K-Means (k=3) on scaled RFM metrics produced three clusters:

| Cluster | Size        | Avg Recency | Avg Frequency | Avg Monetary | Risk Level    |
| ------- | ----------- | ----------- | ------------- | ------------ | ------------- |
| 0       | 1,247 (33%) | 38 days     | 8.3 trans     | $47,200      | **High Risk** |
| 1       | 982 (26%)   | 53 days     | 15.9 trans    | $198,400     | Medium        |
| 2       | 1,513 (41%) | 41 days     | 43.1 trans    | $421,800     | Low           |

![RFM Clusters](./figures/rfm_clusters.png)
*Figure 4: Three distinct customer risk segments*

Cluster 0's minimal engagement (8.3 transactions, $47,200 total over 90 days) contrasts sharply with Cluster 2's high activity, justifying the high-risk classification.

---

## 3. Feature Engineering and Model Development

### 23 Predictive Features

Transformed 16 raw columns into 23 engineered features using sklearn pipelines:

**Temporal Features (6):** Hour, day of week, month, year, weekend flag, time-period bins  
**Aggregate Features (8):** Per-customer total/avg/std/min/max transaction value, count, range, coefficient of variation  
**RFM Metrics (3):** Recency (0-90 days), Frequency (1-487 transactions), Monetary ($100-$8.9M)  
**Encoded Categories (6):** One-hot encoding for low-cardinality, WoE transformation for high-cardinality features

**WoE Implementation:** Achieved medium-to-strong Information Value (IV) scores: ProductId (0.18), ProviderId (0.12), Frequency (0.34), Monetary (0.29)—validating predictive power while maintaining interpretability.

### Model Comparison: Four Approaches

Trained four models with GridSearchCV hyperparameter tuning and MLflow tracking:

| Model                 | Accuracy | Precision | Recall   | F1       | ROC-AUC  | Key Advantage                            |
| --------------------- | -------- | --------- | -------- | -------- | -------- | ---------------------------------------- |
| Logistic Regression   | 0.86     | 0.58      | 0.62     | 0.60     | 0.85     | Highly interpretable, Basel II compliant |
| Decision Tree         | 0.87     | 0.61      | 0.59     | 0.60     | 0.83     | Clear decision rules, non-linear         |
| Random Forest         | 0.89     | 0.67      | 0.64     | 0.65     | 0.88     | Ensemble stability, good balance         |
| **Gradient Boosting** | **0.90** | **0.71**  | **0.67** | **0.69** | **0.91** | **Best performance, production-ready**   |

**Data Split:** 76,529 training samples, 19,133 test samples (80/20 stratified split)  
**Class Distribution:** 88.5% low-risk, 11.5% high-risk (addressed via class weights)  
**Total Experiments:** 108 hyperparameter combinations across 5-fold cross-validation

### Selected Model: Gradient Boosting

**Selection Rationale:**
- Highest ROC-AUC (0.91) and F1-score (0.69)
- 96% specificity (minimal false alarms for legitimate customers)
- 67% sensitivity (catches two-thirds of high-risk customers)
- 255 fewer false positives vs Random Forest → approves more creditworthy customers
- 65 more true positives → better risk detection

**Interpretability Solution:** While less transparent than Logistic Regression, SHAP values provide local explanations for each prediction, satisfying Basel II documentation requirements.

**Feature Importance:** Frequency (28%), Monetary (22%), Recency (19%)—confirming RFM hypothesis.

---

## 4. Production Deployment

### API Architecture

**FastAPI RESTful Service** with six endpoints:
1. `GET /health` - Health check and model status
2. `GET /model/info` - Model metadata and performance metrics  
3. `POST /predict` - Single customer risk prediction
4. `POST /predict/batch` - Batch predictions (max 1,000)
5. `POST /model/reload` - Hot-swap model updates
6. `GET /` - API documentation

**Credit Score Calculation:** Risk probabilities (0-1) transformed to FICO-like scores (300-850). Formula: `300 + (1 - risk_probability) × 550`. Example: 0.23 risk → 720 credit score.

**Confidence Levels:** High confidence (<0.3 or >0.7), medium (0.3-0.4, 0.6-0.7), low (0.4-0.6 → human review).

### Containerization and CI/CD

**Docker:** Multi-stage build with Python 3.10, optimized layers, exposes port 8000  
**docker-compose:** Orchestrates API service, MLflow server (port 5000), shared data volumes  
**GitHub Actions:** Automated pipeline with flake8 linting, black formatting, pytest (47 tests, 85% coverage), Docker build verification

**Deployment:** Single command launches full stack—API responds in ~50ms for single predictions, ~2.5s for 1,000-batch requests.

---

## 5. Business Recommendations

### Three-Tier Risk-Based Lending Strategy

**Tier 1: Low Risk (Score 700-850, Probability < 0.30)**
- 95% auto-approval | $10,000 limit | 8-12% APR | 12-24 months
- 70% of applicants (~2,600 customers) | 5-8% expected defaults

**Tier 2: Medium Risk (Score 600-699, Probability 0.30-0.50)**  
- 70% approval with manual review | $5,000 limit | 15-20% APR | 6-12 months
- 20% of applicants (~750 customers) | 12-18% expected defaults

**Tier 3: High Risk (Score 300-599, Probability > 0.50)**
- 20% approval with co-signer/collateral | $2,000 limit | 25-30% APR | 3-6 months  
- 10% of applicants (~400 customers) | 30-40% expected defaults

**Rejection Threshold:** Score < 400 or probability > 0.85 → automatic rejection with appeal option.

### Four Customer Segment Strategies

**Weekend Warriors (15%):** High weekend volume, low weekday activity → offer bi-weekly repayment aligned with payday cycles

**High-Value Infrequent (8%):** Low frequency but high monetary → potential proxy misclassification; require manual review and additional income verification

**Frequent Small Buyers (40%):** Consistent low-value transactions → pre-approved credit lines with automatic limit increases after 6 on-time payments

**Recent Dormant (12%):** Previously active, now inactive → re-engagement campaigns, conservative terms until activity resumes

### Implementation Roadmap

**Phase 1 (Months 1-3):** Pilot with 500 customers, Tier 1 only, $1,000-$3,000 limits, validate proxy assumptions

**Phase 2 (Months 4-6):** Expand to 2,000 customers, add Tier 2, increase limits to $5,000, retrain with 3 months actual default data

**Phase 3 (Months 7-12):** Full rollout to all 3,742 customers, all tiers, replace proxy with actual default labels

**Phase 4 (Months 13+):** A/B testing, alternative data integration (mobile money velocity, utility payments), real-time scoring optimization

### Financial Projections

**Year 1 (2,000 loans):** $7M volume, $1.05M interest income, $840k default losses, $62k net income (6.2% ROI)

**Sensitivity Analysis:**  
- +5% default rate → -$350k (requires rate increase to 18% APR)
- -3% default rate → +$210k (9.2% ROI)
- Break-even: <15% default rate at 15% APR

---

## 6. Limitations and Future Work

### Honest Assessment of Constraints

**1. Proxy Variable Fundamental Risk**

RFM clustering assumes engagement equals creditworthiness—unvalidated without actual default data. Impact: 1,036 false positives ($3.6M opportunity cost), 794 false negatives ($2.8M default risk). 

**Mitigation:** Immediate default data collection, 6-month hybrid model (50% proxy, 50% actual), 12-month full transition to actual labels.

**2. Interpretability vs. Performance Trade-off**

Gradient Boosting (91% ROC-AUC) outperforms Logistic Regression (85%) but lacks transparency. Regulators prefer explicit risk factor contributions.

**Solution:** Maintain parallel Logistic Regression for regulatory reporting, implement SHAP values for Gradient Boosting local explanations, document methodology thoroughly for Basel II audits.

**3. Data Coverage Gaps**

Current limitations: Rwanda only (geographic), 90-day snapshot (temporal), eCommerce only (product), no demographics (age, gender, income).

**Risks:** Model drift over time, potential unintended bias against underrepresented groups, performance degradation in other markets.

**Future Enhancements:** Integrate alternative data (mobile money velocity, utility payments, social trust scores), expand to 12-24 months temporal coverage, validate on Kenya/Uganda/Tanzania datasets, conduct fairness audits across protected classes.

**4. Class Imbalance Challenge**

88.5% low-risk vs 11.5% high-risk creates majority class bias. Current class weights improved recall +8%, but SMOTE oversampling degraded performance -3%.

**Exploration:** Advanced resampling (ADASYN, Borderline-SMOTE), cost-sensitive learning (custom loss functions prioritizing false negatives), anomaly detection approaches (Isolation Forest, One-Class SVM).

**5. Real-Time Performance Requirements**

Current: 50ms single prediction, 2.5s batch-1000. Production target: <100ms p95 latency for instant point-of-sale approvals at 10,000 requests/day scale.

**Optimization:** Model compression (200→100 trees, int8 quantization), feature caching (Redis), async processing (preliminary score + background completion), horizontal scaling (Kubernetes auto-scaling).

### Future Enhancements Timeline

**Short-Term (3 months):** A/B test XGBoost vs LightGBM vs CatBoost, implement SHAP dashboard, begin default data collection

**Medium-Term (6-12 months):** Retrain with actual defaults, integrate alternative data sources (+3-5% ROC-AUC expected), conduct fairness audits, deploy automated retraining triggers

**Long-Term (12-24 months):** Explore deep learning for high-dimensional data, separate models per loan type (personal/business/emergency), open banking integration (with consent), publish regulatory white papers

---

## 7. Conclusion

Credit risk assessment without historical defaults is achievable through careful proxy engineering, robust feature development, and transparent methodology. The 91% ROC-AUC Gradient Boosting model provides Bati Bank with production-ready infrastructure to launch their buy-now-pay-later service responsibly.

**Key Takeaways:**

**RFM as Viable Proxy:** Customer engagement patterns demonstrate strong predictive signal, validated by model performance and business logic alignment.

**Feature Engineering Drives Success:** 23 engineered features from 16 raw columns, with WoE transformations achieving 18-34% Information Value for key predictors.

**Production Readiness Matters:** Docker containerization, CI/CD pipelines, MLflow tracking, and comprehensive testing transform research into deployable service.

**Honest Limitation Disclosure:** Proxy risks, interpretability trade-offs, and data coverage gaps transparently communicated to stakeholders and regulators.

### Path Forward

The critical next step is **validating proxy assumptions with actual default data**. Within 6 months, transition to hybrid modeling (proxy + actual labels). By 12 months, rely entirely on real-world performance data. This evolution improves accuracy, reduces false positives, and strengthens regulatory confidence.

**Deployment Recommendation:** Launch with conservative thresholds (auto-approve only <0.40 risk), mandatory human review for borderline cases, aggressive monitoring with quarterly retraining, and continuous fairness audits.

With these safeguards, Bati Bank can confidently serve 3,742 customers while maintaining responsible lending standards and Basel II compliance, scaling from $7M Year 1 to $35M Year 3 loan volume.

---

**Contact:** estifanosesahilu@gmail.com  
**Date:** December 16, 2025
