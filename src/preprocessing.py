import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True,
                     encoders=None, scaler=None):
    df = df.copy()

    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        if TARGET in df.columns:
            le_target = LabelEncoder()
            df[TARGET] = le_target.fit_transform(df[TARGET])
            encoders[TARGET] = le_target

        scaler = StandardScaler()
        df[NUMERICAL_COLS] = scaler.fit_transform(df[NUMERICAL_COLS])
        return df, encoders, scaler

    else:
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )
        df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
        return df


def prepare_input(raw_dict: dict, encoders: dict, scaler) -> np.ndarray:
    """Convert a single prediction form dict to a model-ready array."""
    df = pd.DataFrame([raw_dict])
    df = clean_data(df) if "customerID" in df.columns else df
    df = encode_and_scale(df, fit=False, encoders=encoders, scaler=scaler)
    feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS
    return df[feature_cols].values
