import pandas as pd
from src.preprocessing import (
    load_data, clean_data, encode_and_scale,
    CATEGORICAL_COLS, NUMERICAL_COLS, TARGET,
)
from src.model import (
    split_data, train_random_forest, evaluate_model, save_artefacts,
)


def main():
    print(" ChurnScope Model Training\n")

    # 1. Load & clean
    print("[1/5] Loading data")
    df = load_data("data/telco_churn.csv")
    df = clean_data(df)
    print(f"     {len(df)} rows loaded. Churn rate: {df[TARGET].value_counts(normalize=True)['Yes']:.1%}")

    # 2. Encode & scale
    print("[2/5] Encoding and scaling features")
    df_enc, encoders, scaler = encode_and_scale(df, fit=True)

    # 3. Split
    print("[3/5] Splitting train / test (80 / 20)")
    feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS
    X = df_enc[feature_cols].values
    y = df_enc[TARGET].values
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Train
    print("[4/5] Training Random Forest (200 trees)")
    clf = train_random_forest(X_train, y_train)

    # 5. Evaluate & save
    print("[5/5] Evaluating and saving artefacts")
    metrics = evaluate_model(clf, X_test, y_test, feature_cols)
    save_artefacts(clf, encoders, scaler, metrics)

    print("\n--- Results ---")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nDone. Run `flask run` or `python app.py` to start the web app.")


if __name__ == "__main__":
    main()
