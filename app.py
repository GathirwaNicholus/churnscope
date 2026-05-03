import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.model import load_artefacts
from src.preprocessing import (
    prepare_input, CATEGORICAL_COLS, NUMERICAL_COLS,
    load_data, clean_data,
)

app = Flask(__name__)

# Load model artefacts once at startup 
clf, encoders, scaler, metrics = load_artefacts()

# Load raw data for dashboard EDA charts 
_raw_df = None


def get_raw_df():
    global _raw_df
    if _raw_df is None:
        _raw_df = clean_data(load_data("data/telco_churn.csv"))
    return _raw_df


# Routes

@app.route("/")
def index():
    df = get_raw_df()
    summary = {
        "total_customers": len(df),
        "churn_count": int((df["Churn"] == "Yes").sum()),
        "churn_rate": round((df["Churn"] == "Yes").mean() * 100, 1),
        "avg_tenure": round(df["tenure"].mean(), 1),
        "avg_monthly": round(df["MonthlyCharges"].mean(), 2),
        "model_accuracy": round(metrics["accuracy"] * 100, 1),
        "roc_auc": round(metrics["roc_auc"], 4),
    }
    return render_template("index.html", summary=summary)


@app.route("/dashboard")
def dashboard():
    df = get_raw_df()

    # Churn by contract type
    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(1)
        .to_dict()
    )

    # Churn by internet service
    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(1)
        .to_dict()
    )

    # Tenure distribution (churned vs retained) — bucketed
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)
    tenure_dist = (
        df.groupby(["tenure_bucket", "Churn"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"No": "Retained", "Yes": "Churned"})
        .to_dict()
    )
    tenure_dist = {
        "labels": labels,
        "churned": [int(tenure_dist["Churned"].get(l, 0)) for l in labels],
        "retained": [int(tenure_dist["Retained"].get(l, 0)) for l in labels],
    }

    # Monthly charges distribution (churned vs retained) — bucketed
    df["charge_bucket"] = pd.cut(
        df["MonthlyCharges"], bins=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    charge_dist = (
        df.groupby(["charge_bucket", "Churn"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"No": "Retained", "Yes": "Churned"})
    )
    charge_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    charge_data = {
        "labels": charge_labels,
        "churned": [int(charge_dist.get("Churned", {}).get(l, 0)) for l in charge_labels],
        "retained": [int(charge_dist.get("Retained", {}).get(l, 0)) for l in charge_labels],
    }

    # Feature importances (from model)
    feat_imp = metrics["feature_importances"]

    # Confusion matrix
    cm = metrics["confusion_matrix"]

    # ROC curve
    roc = metrics["roc_curve"]

    chart_data = {
        "contract_churn": contract_churn,
        "internet_churn": internet_churn,
        "tenure_dist": tenure_dist,
        "charge_data": charge_data,
        "feat_imp": feat_imp,
        "cm": cm,
        "roc": roc,
    }

    return render_template(
        "dashboard.html",
        metrics=metrics,
        chart_data=json.dumps(chart_data),
    )


def _f(val, default=0.0):
    """Safely converting a form string value to float, returning default for blank input."""
    try:
        return float(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _i(val, default=0):
    """Safely converting a form string value to int."""
    try:
        return int(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form_data = {}
    error = None

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            raw = {
                "gender": form_data.get("gender", "Male"),
                "SeniorCitizen": _i(form_data.get("SeniorCitizen"), 0),
                "Partner": form_data.get("Partner", "No"),
                "Dependents": form_data.get("Dependents", "No"),
                "tenure": _f(form_data.get("tenure"), 0.0),
                "PhoneService": form_data.get("PhoneService", "Yes"),
                "MultipleLines": form_data.get("MultipleLines", "No"),
                "InternetService": form_data.get("InternetService", "DSL"),
                "OnlineSecurity": form_data.get("OnlineSecurity", "No"),
                "OnlineBackup": form_data.get("OnlineBackup", "No"),
                "DeviceProtection": form_data.get("DeviceProtection", "No"),
                "TechSupport": form_data.get("TechSupport", "No"),
                "StreamingTV": form_data.get("StreamingTV", "No"),
                "StreamingMovies": form_data.get("StreamingMovies", "No"),
                "Contract": form_data.get("Contract", "Month-to-month"),
                "PaperlessBilling": form_data.get("PaperlessBilling", "Yes"),
                "PaymentMethod": form_data.get("PaymentMethod", "Electronic check"),
                "MonthlyCharges": _f(form_data.get("MonthlyCharges"), 0.0),
                "TotalCharges": _f(form_data.get("TotalCharges"), 0.0),
            }

            X = prepare_input(raw, encoders, scaler)
            prob = float(clf.predict_proba(X)[0][1])
            prediction = "Churn" if prob >= 0.5 else "No Churn"
            risk_level = "Low" if prob < 0.35 else ("Medium" if prob < 0.65 else "High")

            result = {
                "prediction": prediction,
                "probability": round(prob * 100, 1),
                "risk_level": risk_level,
            }
        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template("predict.html", result=result, form_data=form_data, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic predictions."""
    data = request.get_json(force=True)
    try:
        X = prepare_input(data, encoders, scaler)
        prob = float(clf.predict_proba(X)[0][1])
        return jsonify({
            "churn_probability": round(prob, 4),
            "prediction": "Churn" if prob >= 0.5 else "No Churn",
            "risk_level": "Low" if prob < 0.35 else ("Medium" if prob < 0.65 else "High"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
