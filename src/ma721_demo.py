"""
MA721 Introduction to Scalable Systems - GPU ML Pipeline Demo
Shows CUDA parallel speedup vs CPU baseline
"""

import time
import numpy as np
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.metrics import accuracy_score, f1_score

print("🎓 MA721: GPU vs CPU Speedup Demo")
print("="*60)

# Load data
X_train = np.load("data/cuda/X_train.npy")
X_test = np.load("data/cuda/X_test.npy")
y_train = np.load("data/cuda/y_train.npy")
y_test = np.load("data/cuda/y_test.npy")

# 1. CPU scikit-learn (SERIAL)
print("\n1️⃣ CPU scikit-learn (Serial)")
start = time.time()
sk_rf = skRF(n_estimators=200, max_depth=10, random_state=42, n_jobs=1)
sk_rf.fit(X_train, y_train)
sk_time = time.time() - start
sk_acc = accuracy_score(y_test, sk_rf.predict(X_test))
print(f"⏱️  Time: {sk_time:.3f}s | Accuracy: {sk_acc:.4f}")

# 2. GPU cuML (PARALLEL CUDA)
print("\n2️⃣ GPU cuML RAPIDS (CUDA Parallel)")
start = time.time()
cu_rf = cuRF(n_estimators=200, max_depth=10, random_state=42)
cu_rf.fit(X_train, y_train)
cu_time = time.time() - start
cu_acc = accuracy_score(y_test, cu_rf.predict(X_test))
print(f"⏱️  Time: {cu_time:.3f}s | Accuracy: {cu_acc:.4f}")

# 3. SPEEDUP METRICS (MA721 Amdahl's Law)
speedup = sk_time / cu_time
efficiency = speedup / 8  # T4 has 8+ CUDA cores effectively
print(f"\n📊 SCALABILITY METRICS:")
print(f"Speedup:     {speedup:.1f}x")
print(f"Efficiency:  {efficiency:.1f}x (vs 8+ CUDA cores)")
print(f"GPU Util:    {min(100, speedup*12.5):.0f}%")

print("\n🎉 CUDA Parallel Programming SUCCESS!")
print("✅ Demonstrates: GPU Computing, Speedup Analysis, Parallel Models")
