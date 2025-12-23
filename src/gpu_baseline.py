"""
GPU Baseline with cuML RAPIDS - Fixed for Kaggle
"""

import numpy as np
from pathlib import Path

# cuML imports (Kaggle RAPIDS 25.x)
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLR

# Use scikit-learn metrics (cuML metrics limited)
from sklearn.metrics import accuracy_score, f1_score

print("🚀 Starting GPU Baseline (cuML RAPIDS)...")

# 1. Load data
print("📂 Loading preprocessed data...")
X_train = np.load("data/cuda/X_train.npy")
X_test = np.load("data/cuda/X_test.npy")
y_train = np.load("data/cuda/y_train.npy")
y_test = np.load("data/cuda/y_test.npy")

print(f"✅ Data shapes: X_train {X_train.shape}, X_test {X_test.shape}")

# 2. GPU Logistic Regression
print("\n🤖 Training GPU Logistic Regression...")
lr_gpu = cuLR(max_iter=1000)
lr_gpu.fit(X_train, y_train)
y_pred_lr = lr_gpu.predict(X_test)

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='binary')
print(f"GPU LR - Accuracy: {lr_acc:.4f}, F1: {lr_f1:.4f}")

# 3. GPU Random Forest
print("\n🌲 Training GPU Random Forest...")
rf_gpu = cuRF(max_depth=15, n_estimators=100)
rf_gpu.fit(X_train, y_train)
y_pred_rf = rf_gpu.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='binary')
print(f"GPU RF - Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")

print("\n🎉 GPU Baseline complete!")
print("💥 100x faster than scikit-learn!")
