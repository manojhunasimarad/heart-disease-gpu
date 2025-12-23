"""
Load model and predict new patient
"""

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score

# Load model
model = joblib.load("data/models/best_gpu_rf.joblib")
X_test = np.load("data/cuda/X_test.npy")
y_test = np.load("data/cuda/y_test.npy")

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')

print(f"✅ Loaded model - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
print("🎯 Ready for new patients!")
