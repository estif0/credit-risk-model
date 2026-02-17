# Credit Risk Model - Presentation Slides Content

**Target Audience:** Finance Recruiters & Hiring Managers  
**Business Value:** Risk mitigation, loan portfolio optimization, real-time decisioning.

---

## Slide 1: Title
**Title:** Production-Grade Credit Risk Scoring for BNPL  
**Subtitle:** Enhancing Bati Bank's Lending Capability with Machine Learning  
**Presenter:** Estifanose Sahilu

---

## Slide 2: The Business Problem
- **Context:** Bati Bank's Buy-Now-Pay-Later (BNPL) service.
- **Problem:** Need for automated, reliable credit risk assessment to reduce defaults while expanding access.
- **Goal:** Predict defaults before they happen using historical transaction data.

---

## Slide 3: Solution Architecture
- **Data Layer:** RFM (Recency, Frequency, Monetary) Analysis & Feature Engineering.
- **ML Layer:** Multiple models (Logistic Regression, Random Forest, GBDT) tracked via MLflow.
- **API Layer:** High-performance FastAPI with rate limiting and comprehensive logging.
- **Frontend Layer:** Interactive Streamlit Dashboard for business stakeholders.

---

## Slide 4: Key Technical Features
- **100% Test Coverage:** 134 automated tests ensuring system reliability.
- **Real-time Scoring:** Instant credit score calculation (300-850 range).
- **Batch Processing:** Ability to process thousands of transactions via CSV upload.
- **Monitoring:** Request logging and rate limiting for production security.

---

## Slide 5: Model Performance
- **Accuracy:** High AUC score indicating excellent separation between risk categories.
- **Interpretability:** Feature importance analysis (e.g., Monetary value and Frequency as key predictors).
- **Compliance:** Documentation and logging aligned with Basel II standards.

---

## Slide 6: Business Value & ROI
- **Reduced Defaults:** Early detection of high-risk transactions.
- **Efficiency:** Automation of the loan approval workflow.
- **Scalability:** System capable of handling increasing transaction volume via REST API.
- **Portfolio Health:** Data-driven decisions leads to more stable lending portfolios.

---

## Slide 7: Visual Showcase (Screenshots)
- *Include "Dashboard Overview" and "Single Prediction" screenshots from `reports/figures/`*
- Highlighting the "Credit Score" and "Risk Probability" outputs.

---

## Slide 8: Conclusion & Future Scope
- **Conclusion:** A robust, production-ready system bridging the gap between data science and business operations.
- **Future:** Integration with real-time financial data providers and advanced deep learning models.

---

## Slide 9: Q&A
**Thank You!**  
*Contact: [Your Email/LinkedIn]*
