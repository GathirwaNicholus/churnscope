# ChurnScope

A machine learning web application that predicts customer churn for a telecom company. Built with Python, Scikit-learn, and Flask.

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train_model.py
python app.py
```

Open http://127.0.0.1:5000

## Stack

Python · Scikit-learn (Random Forest) · Flask · Chart.js

## Pages

- **Overview** — KPI summary cards
- **Analytics** — EDA charts and model performance (Confusion Matrix, ROC Curve, Feature Importances)
- **Predict** — Single customer churn risk form

## API

```
POST /api/predict  →  { "churn_probability": 0.78, "prediction": "Churn", "risk_level": "High" }
```
