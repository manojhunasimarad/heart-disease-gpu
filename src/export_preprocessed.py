"""
Export preprocessed data for GPU (cuML RAPIDS)
Saves X_train.npy, X_test.npy, y_train.npy, y_test.npy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.pipeline import Pipeline
import glob
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

print("🚀 Exporting preprocessed data for GPU...")

# 1. Load & preprocess (same as cpu_baseline)
possible_paths = [
    "/kaggle/input/heart-disease-data/heart_disease_uci.csv",
    "data/raw/heartdiseaseuci.csv",
    "/kaggle/input/heart-disease-uci/heart.csv"
]

datapath = None
for path in possible_paths:
    if Path(path).exists():
        datapath = Path(path)
        break

print(f"✅ Found: {datapath}")
df = pd.read_csv(datapath)

numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

X = df.drop(columns=['num'])
y = (df['num'] > 0).astype(int)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Transform
print("🔄 Preprocessing...")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# 2. Save as NumPy arrays for cuML
os.makedirs("data/cuda", exist_ok=True)
np.save("data/cuda/X_train.npy", X_train_proc)
np.save("data/cuda/X_test.npy", X_test_proc)
np.save("data/cuda/y_train.npy", y_train.values)
np.save("data/cuda/y_test.npy", y_test.values)

print(f"✅ Saved:")
print(f"  X_train: {X_train_proc.shape}")
print(f"  X_test:  {X_test_proc.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test:  {y_test.shape}")
print("🎉 GPU data ready!")
