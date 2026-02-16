# Credit Risk Model - Video Demo Script

**Duration:** 3-5 minutes  
**Date:** February 16, 2026

---

## Pre-Recording Checklist

- [ ] Start API server: `uvicorn src.api.main:app --port 9000`
- [ ] Start Streamlit dashboard: `streamlit run src/dashboard/app.py`
- [ ] Open browser tabs:
  - Dashboard: http://localhost:8501
  - API Swagger UI: http://localhost:9000/docs
- [ ] Prepare sample CSV file for batch demo (if not already available)
- [ ] Close unnecessary applications
- [ ] Set screen resolution to 1920x1080 (or 1280x720)

---

## Recording Script

### Introduction (30 seconds)

> "Hello, I'm Estifanose Sahilu, and this is my Credit Risk Scoring Model for Bati Bank's buy-now-pay-later service. This production-grade system uses machine learning and RFM analytics to predict credit risk in real-time. Let me walk you through the key features."

---

### Part 1: Dashboard Overview (45 seconds)

**Action:** Navigate to Overview page

> "The dashboard provides a comprehensive view of the system. Here you can see:
> - System status showing the application is running
> - Active model information - we're using a Logistic Regression model trained on February 11th
> - Key performance metrics: 100% success rate and perfect ROC-AUC score of 1.0
> - This gives business users immediate visibility into model health."

---

### Part 2: Single Prediction Demo (60 seconds)

**Action:** Navigate to Single Prediction page

> "For individual assessments, loan officers can use the Single Prediction interface. Let me demonstrate with two examples."

**Low-Risk Example:**
- Amount: 5000
- Total Transaction Amount: 50000
- Transaction Count: 100
- Average Transaction Amount: 500
- Click "Predict Risk"

> "For a customer with consistent transaction history - 100 transactions averaging 500 birr - the model predicts LOW risk with 99.99% confidence and assigns a credit score of 820, which is excellent."

**High-Risk Example:**
- Amount: 50000
- Total Transaction Amount: 5000
- Transaction Count: 2
- Average Transaction Amount: 2500
- Click "Predict Risk"

> "In contrast, a customer requesting 50,000 birr with only 2 transactions shows HIGH risk with 99.99% probability and a credit score of 305, indicating they should be declined or offered reduced credit."

---

### Part 3: Batch Analysis (45 seconds)

**Action:** Navigate to Batch Analysis page

> "For bulk processing, the Batch Analysis feature allows uploading CSV files with multiple customer records. The system processes all transactions and provides downloadable results for integration into existing loan approval workflows. This is critical for processing hundreds of applications daily."

**Action:** Show CSV upload interface (don't need to actually upload unless you have a sample ready)

---

### Part 4: Model Performance (30 seconds)

**Action:** Navigate to Model Performance page

> "The Model Performance page provides transparency into how the model makes decisions. You can see:
> - ROC-AUC curves showing model accuracy
> - Feature importance rankings - which factors matter most
> - This is essential for regulatory compliance under Basel II requirements."

---

### Part 5: API Demonstration (30 seconds)

**Action:** Switch to browser tab with http://localhost:9000/docs

> "The system also exposes a RESTful API for integration with other systems. The Swagger UI shows all available endpoints:
> - /predict for single predictions
> - /predict/batch for bulk processing
> - /model/info for model metadata
> - All endpoints are fully tested with 100% coverage."

**Action:** Expand the /predict endpoint to show the request/response schema

---

### Conclusion (20 seconds)

> "This production-ready system demonstrates technical excellence with 134 passing tests, comprehensive monitoring, and a business-friendly interface. It's designed to help Bati Bank reduce default rates while providing credit access to qualified customers. Thank you for watching."

---

## Post-Recording Steps

1. Review the recording for audio/video quality
2. Trim any unnecessary parts (keep under 5 minutes)
3. Export as MP4 format
4. Save as `reports/credit_risk_demo.mp4`
5. Test playback to ensure quality

---

## Recording Software Commands

### Option 1: SimpleScreenRecorder (Recommended)
```bash
# Install if not available
sudo apt install simplescreenrecorder

# Launch
simplescreenrecorder
```

**Settings:**
- Video input: Record entire screen (or select window)
- Audio: Enable microphone
- Output: MP4 (H.264 + AAC)
- Frame rate: 30 fps
- Quality: High

### Option 2: OBS Studio
```bash
# Install if not available
sudo apt install obs-studio

# Launch
obs
```

**Settings:**
- Scene: Display Capture
- Audio: Microphone/Aux
- Output: Recording → MP4 format
- Video bitrate: 2500 Kbps

---

## Troubleshooting

**If API doesn't start:**
```bash
cd /home/voldi/Projects/ai-ml/credit-risk-model
source venv/bin/activate
uvicorn src.api.main:app --port 9000
```

**If Dashboard doesn't start:**
```bash
cd /home/voldi/Projects/ai-ml/credit-risk-model
source venv/bin/activate
streamlit run src/dashboard/app.py
```

**If MLflow model not found:**
```bash
# Check available models
ls mlruns/
# The dashboard will auto-load the latest model
```
