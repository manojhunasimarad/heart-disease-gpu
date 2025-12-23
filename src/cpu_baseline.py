# src/cpu_baseline.py
"""
CPU baseline for heart_disease_uci.csv using scikit-learn.

Run from repo root:
    python -m src.cpu_baseline
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


DATA_PATH = "data/heart_disease_uci.csv"


def main() -> None:
    print("CPU baseline: Logistic Regression on heart_disease_uci.csv")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    numeric_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    X = df.drop(columns=["num"])
    y = (df["num"] > 0).astype(np.int32)

    # Preprocessing pipelines (same as notebook)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Full pipeline: preprocessing + Logistic Regression
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t1 = time.perf_counter()

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("\nMetrics on test split:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1-score : {f1:.3f}")

    print(f"\nTraining time (CPU, LogisticRegression): {t1 - t0:.4f} seconds")


if __name__ == "__main__":
    main()
