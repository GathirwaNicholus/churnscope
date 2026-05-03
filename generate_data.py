
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 2000

def generate_dataset(n=N):
    ids = [f"C-{str(i).zfill(5)}" for i in range(1, n + 1)]
    gender = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = np.random.randint(0, 73, n)

    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        np.random.choice(["Yes", "No"], n, p=[0.42, 0.58]),
    )

    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    def internet_feature(internet, p_yes=0.45):
        return np.where(
            internet == "No",
            "No internet service",
            np.random.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes]),
        )

    online_security = internet_feature(internet_service, 0.29)
    online_backup = internet_feature(internet_service, 0.34)
    device_protection = internet_feature(internet_service, 0.34)
    tech_support = internet_feature(internet_service, 0.29)
    streaming_tv = internet_feature(internet_service, 0.38)
    streaming_movies = internet_feature(internet_service, 0.39)

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = np.random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    monthly_charges = np.where(
        internet_service == "No",
        np.random.uniform(18, 35, n),
        np.where(
            internet_service == "DSL",
            np.random.uniform(40, 75, n),
            np.random.uniform(60, 110, n),
        ),
    ).round(2)

    total_charges = (tenure * monthly_charges + np.random.uniform(-10, 10, n)).round(2)
    total_charges = np.where(tenure == 0, np.nan, total_charges)

    # Churn probability — higher for month-to-month, fiber, short tenure
    churn_prob = (
        0.10
        + 0.25 * (contract == "Month-to-month").astype(float)
        + 0.15 * (internet_service == "Fiber optic").astype(float)
        - 0.15 * (tenure > 36).astype(float)
        + 0.10 * (payment_method == "Electronic check").astype(float)
        + 0.05 * (senior == 1).astype(float)
        - 0.05 * (partner == "Yes").astype(float)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.85)
    churn = np.where(np.random.random(n) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    out = "data/telco_churn.csv"
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}  ({len(df)} rows, {df['Churn'].value_counts()['Yes']} churned)")
