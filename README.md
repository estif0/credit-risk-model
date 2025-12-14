# Credit Risk Model

## Credit Scoring Business Understanding

### Basel II and Model Interpretability

The Basel II Capital Accord establishes international standards for banking supervision, with a strong emphasis on **quantitative risk measurement** and **model validation**. For our credit risk model, Basel II's requirements significantly influence our design decisions:

**Key Basel II Implications:**

1. **Model Transparency**: Basel II mandates that financial institutions must understand and document their risk measurement approaches. This requirement drives us toward **interpretable models** where we can explain:
   - Which features contribute to risk assessments
   - How different customer behaviors impact credit decisions
   - The rationale behind risk probability estimates

2. **Validation Requirements**: Our model must be rigorously tested and validated with clear documentation of:
   - Model assumptions and limitations
   - Performance metrics across different customer segments
   - Back-testing procedures to verify predictive accuracy
   - Ongoing monitoring processes

3. **Risk Quantification**: Basel II requires precise risk probability estimates to calculate regulatory capital requirements. Our model must provide:
   - Probability of Default (PD) estimates
   - Confidence intervals for predictions
   - Calibration across the entire risk spectrum

**Impact on Our Approach**: These requirements favor models with **clear decision boundaries** and **traceable feature contributions**, making documentation and explainability as important as predictive performance.

---

### The Necessity and Risks of Proxy Variables

#### Why Proxy Variables Are Necessary

In traditional credit scoring, models are trained on historical default data—actual records of customers who failed to repay loans. However, for our buy-now-pay-later service, we face a **cold start problem**:

- **No Historical Default Data**: As a new service, we have no direct records of loan defaults
- **Only Transactional Data**: We possess eCommerce behavioral data (purchases, amounts, timing, channels)
- **Regulatory Timeline**: We cannot wait years to accumulate default history before launching the service

**Our Solution: RFM-Based Proxy Variable**

We create a proxy for credit risk by analyzing customer engagement patterns through **RFM (Recency, Frequency, Monetary) analysis**:

- **Recency**: Days since last transaction (disengaged customers may be higher risk)
- **Frequency**: Number of transactions (active customers show commitment)
- **Monetary**: Total transaction value (financial capacity indicator)

Using K-Means clustering, we segment customers into risk categories, assuming that **least engaged customers** (high recency, low frequency/monetary) represent **higher credit risk**.

#### Business Risks of Proxy-Based Predictions

While necessary, this proxy approach introduces significant risks:

1. **Correlation vs Causation**:
   - Low engagement doesn't necessarily mean inability to repay
   - A customer might have low transaction frequency but high creditworthiness
   - We're assuming engagement patterns predict financial reliability—an untested hypothesis

2. **Potential for Discrimination**:
   - May unfairly penalize new customers or those with changing purchase patterns
   - Could exclude creditworthy individuals based on shopping behavior rather than financial capacity
   - Risk of systematic bias against certain customer segments

3. **Model Drift Over Time**:
   - As we accumulate actual default data, the relationship between engagement and default may prove weaker than assumed
   - The model will require retraining with ground truth labels
   - Initial predictions may have limited real-world validity

4. **Regulatory Scrutiny**:
   - Basel II requires models to be based on sound statistical evidence
   - Our proxy approach may face challenges in regulatory validation
   - We must clearly disclose the experimental nature of this methodology

5. **Business Impact**:
   - False positives (rejecting good customers) result in lost revenue
   - False negatives (approving bad customers) lead to defaults and losses
   - Without ground truth, we cannot properly calibrate this trade-off

**Mitigation Strategy**: We must implement rigorous monitoring, collect actual default data from day one, and plan for model retraining once sufficient ground truth is available (typically 12-24 months).

---

### Model Trade-offs: Simplicity vs Complexity

Choosing between simple, interpretable models and complex, high-performance models involves critical trade-offs, especially in a regulated financial context.

#### Simple Models: Logistic Regression with WoE

**Advantages:**
- **High Interpretability**: Each coefficient clearly shows feature impact on risk
- **Weight of Evidence (WoE)** transformation provides:
  - Monotonic relationships between features and target
  - Clear binning strategies that business users can understand
  - Information Value (IV) metrics for feature selection
- **Regulatory Acceptance**: Well-understood methodology with decades of use in banking
- **Fast Inference**: Minimal computational requirements for real-time scoring
- **Easy Debugging**: Simple to diagnose and fix when issues arise
- **Stakeholder Communication**: Non-technical executives can understand model decisions

**Disadvantages:**
- **Linear Assumptions**: Cannot capture complex non-linear relationships or feature interactions
- **Lower Predictive Power**: May miss subtle patterns that indicate risk
- **Feature Engineering Intensive**: Requires manual creation of interaction terms and polynomial features
- **Limited with Complex Data**: Struggles with high-dimensional feature spaces

**Basel II Fit**: Excellent—interpretability and transparency align perfectly with regulatory requirements.

#### Complex Models: Gradient Boosting (XGBoost, LightGBM)

**Advantages:**
- **Superior Predictive Performance**: Captures non-linear relationships and complex feature interactions
- **Automatic Feature Engineering**: Learns optimal feature combinations without manual specification
- **Handles Missing Data**: Built-in mechanisms for missing value treatment
- **Robust to Outliers**: Less sensitive to extreme values than linear models
- **Feature Importance**: Provides SHAP values and gain metrics for interpretation
- **Ensemble Learning**: Reduces overfitting through multiple weak learners

**Disadvantages:**
- **Black Box Nature**: Difficult to explain individual predictions to regulators or customers
- **Regulatory Challenges**: May face scrutiny under Basel II's transparency requirements
- **Overfitting Risk**: Can memorize training data, especially with limited samples
- **Computational Complexity**: Requires more resources for training and inference
- **Hyperparameter Sensitivity**: Performance heavily depends on tuning
- **Harder to Maintain**: More difficult to debug and update as business rules change

**Basel II Fit**: Challenging—while powerful, the lack of transparency may not satisfy regulatory validation requirements without extensive documentation and post-hoc explanation techniques (SHAP, LIME).

#### Our Recommended Approach: Hybrid Strategy

Given our context (proxy-based target, regulatory requirements, new service), we recommend:

1. **Start with Logistic Regression + WoE**:
   - Establish an interpretable baseline
   - Build trust with regulators and stakeholders
   - Understand which features truly matter
   - Document clear decision rules

2. **Develop Gradient Boosting in Parallel**:
   - Compare performance metrics
   - Use SHAP values for model explanation
   - Validate that complex model findings align with business intuition

3. **Implement Model Governance**:
   - If deploying Gradient Boosting, create extensive documentation
   - Implement human-in-the-loop review for borderline cases
   - Monitor model decisions for bias and drift
   - Maintain Logistic Regression as a fallback and validation tool

4. **Plan for Evolution**:
   - As we collect actual default data, reassess model choice
   - Consider ensemble methods that combine interpretability and performance
   - Invest in explainable AI tools (SHAP, LIME) to bridge the gap

**Conclusion**: In a regulated environment with a proxy target variable, we must prioritize **interpretability and validation** over marginal performance gains. Starting with simple models allows us to build a solid foundation, while complex models can be introduced later with proper governance frameworks once we have ground truth data to validate their predictions.
