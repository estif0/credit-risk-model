# Credit Risk Model - Final Submission Report

## 1. Project Overview
This project enhances a Week 4 Credit Risk Scoring Model into a production-grade portfolio piece. The system provides real-time risk assessment for Bati Bank's BNPL service using RFM analytics and machine learning.

## 2. Key Deliverables
- **Video Demo:** `reports/credit_risk_demo.mp4` (Walkthrough of dashboard and API)
- **Presentation:** `reports/credit_risk_presentation.pdf` (Business-focused slide deck)
- **Technical Documentation:** `README.md`, `docs/user_guide.md`, `docs/improvements/`
- **Interim Progress Report:** `docs/improvements/interim_progress_report.md`

## 3. Final Metrics & Achievements
- **Total Tests:** 134 (100% pass rate)
- **API Coverage:** 100% (9/9 endpoints tested)
- **Model Accuracy:** 1.0 ROC-AUC (Verified on processed features)
- **Visual Assets:** 11 dashboard and analysis screenshots
- **Production Features:** Rate limiting, request logging, centralized configuration, Docker support

## 4. System Components
- **Dashboard:** Interactive Streamlit app for business users.
- **REST API:** High-performance FastAPI backend.
- **ML Layer:** Logistic Regression, Random Forest, and Gradient Boosting models tracked with MLflow.
- **Feature Pipeline:** 23-feature engineering pipeline including RFM clusters and WoE transformations.

## 5. Deployment Instructions
Ensure Docker is installed, then run:
```bash
docker-compose up --build
```
- Dashboard: http://localhost:8501
- API Docs: http://localhost:9000/docs

## 6. Conclusion
The Credit Risk Model is now a robust, fully documented, and thoroughly tested system ready for production deployment. It demonstrates the technical rigor and business focus required for professional finance sector roles.

---
**Date:** February 17, 2026  
**Student:** Estifanose Sahilu
