"""
CPU Baseline for Heart Disease Prediction - Auto Data Finder
Works in Kaggle/Colab/local environments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("🚀 Starting CPU Baseline...")
print("Current dir:", os.getcwd())

# 1. SMART DATA LOADING - finds CSV automatically
print("\n📊 Finding heart disease data...")

# Try multiple common paths (YOUR PATH ADDED!)
possible_paths = [
    "data/raw/heartdiseaseuci.csv",
    "/kaggle/input/heart-disease-data/heart_disease_uci.csv",  # ← YOUR PATH HERE
    "/kaggle/input/heart-disease-data/heartdiseaseuci.csv",
    "/kaggle/input/heart-disease-uci/heart.csv", 
    "/kaggle/input/heart-disease-uci/heartdiseaseuci.csv",
    "../input/heart-disease-data/*.csv",
    "heartdiseaseuci.csv",
    "heart.csv",
    "/kaggle/input/heart-disease-data/*.csv"
]

datapath = None
for path in possible_paths:
    if '*' in path:
        matches = glob.glob(path)
        if matches:
            datapath = Path(matches[0])
            break
    else:
        if Path(path).exists():
            datapath = Path(path)
            break

if datapath is None:
    print("❌ Dataset not found! Common locations checked.")
    print("Please add dataset to Kaggle or place CSV in current directory.")
    exit(1)

print(f"✅ Found: {datapath}")
df = pd.read_csv(datapath)
print(f"Loaded {len(df)} rows x {len(df.columns)} cols")
print("Shape:", df.shape)

# 2. Features (same as your notebook)
numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

X = df.drop(columns=['num'])
y = (df['num'] > 0).astype(int)

# 3. Preprocessing pipeline (same as notebook)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Train Logistic Regression
print("\n🤖 Training Logistic Regression...")
clf = Pipeline(steps=[('preprocess', preprocessor),
                      ('model', LogisticRegression(max_iter=1000))])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 6. Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"\n✅ Logistic Regression Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {f1:.4f}")
print("🎉 CPU Baseline complete!")
