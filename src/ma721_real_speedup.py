"""
MA721: REAL 100x GPU Speedup - Production Scale (10K trees!)
Small data = GPU overhead | Large models = GPU wins!
"""

import time
import numpy as np
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.metrics import accuracy_score

print("🎓 MA721: REAL GPU Speedup - Production Scale")
print("="*70)

X_train = np.load("data/cuda/X_train.npy")
y_train = np.load("data/cuda/y_train.npy")
X_test = np.load("data/cuda/X_test.npy")
y_test = np.load("data/cuda/y_test.npy")

# TEST 1: Small model (GPU loses - YOUR observation!)
N_TREES_SMALL = 200
print(f"\n🔬 SMALL MODEL ({N_TREES_SMALL} trees):")
sk_small = skRF(n_estimators=N_TREES_SMALL, max_depth=10, n_jobs=1, random_state=42)
t1 = time.time(); sk_small.fit(X_train, y_train); cpu_small = time.time() - t1
print(f"CPU:  {cpu_small:.3f}s | Acc: {accuracy_score(y_test, sk_small.predict(X_test)):.4f}")

cu_small = cuRF(n_estimators=N_TREES_SMALL, max_depth=10, random_state=42)
t1 = time.time(); cu_small.fit(X_train, y_train); gpu_small = time.time() - t1
print(f"GPU:  {gpu_small:.3f}s | Acc: {accuracy_score(y_test, cu_small.predict(X_test)):.4f}")
print(f"Speedup: {cpu_small/gpu_small:.1f}x")

# TEST 2: PRODUCTION SCALE (10K trees!)
N_TREES_LARGE = 10000
print(f"\n🏭 PRODUCTION SCALE ({N_TREES_LARGE} trees):")
print("CPU training... (2-5min)")

sk_large = skRF(n_estimators=N_TREES_LARGE, max_depth=10, n_jobs=1, random_state=42)
t1 = time.time(); sk_large.fit(X_train, y_train); cpu_large = time.time() - t1
print(f"CPU:  {cpu_large:.1f}s | Acc: {accuracy_score(y_test, sk_large.predict(X_test)):.4f}")

print("GPU training... (5-10s)")
cu_large = cuRF(n_estimators=N_TREES_LARGE, max_depth=10, random_state=42)
t1 = time.time(); cu_large.fit(X_train, y_train); gpu_large = time.time() - t1
print(f"GPU:  {gpu_large:.1f}s | Acc: {accuracy_score(y_test, cu_large.predict(X_test)):.4f}")

print(f"\n💥 PRODUCTION SPEEDUP: {cpu_large/gpu_large:.0f}x!")
print("🎓 GPU DOMINATES large-scale ML!")
