# src/export_preprocessed.py
"""
Export preprocessed design matrix X and labels y for CUDA logistic regression.

Run from repo root:
    python -m src.export_preprocessed
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "data/heart_disease_uci.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    numeric_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    X = df.drop(columns=["num"])
    y = (df["num"] > 0).astype(np.float32)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    X_train_t = X_train_t.astype(np.float32)
    X_test_t = X_test_t.astype(np.float32)

    np.save("data/X_train_cuda.npy", X_train_t)
    np.save("data/y_train_cuda.npy", y_train.to_numpy(dtype=np.float32))
    np.save("data/X_test_cuda.npy", X_test_t)
    np.save("data/y_test_cuda.npy", y_test.to_numpy(dtype=np.float32))

    print("Saved:")
    print("  data/X_train_cuda.npy", X_train_t.shape)
    print("  data/y_train_cuda.npy", y_train.shape)
    print("  data/X_test_cuda.npy", X_test_t.shape)
    print("  data/y_test_cuda.npy", y_test.shape)

if __name__ == "__main__":
    main()
