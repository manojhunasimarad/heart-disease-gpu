# Same as gpu_baseline.py but with tuned params
rf_gpu = cuRF(max_depth=10, n_estimators=200, random_state=42)
# Expect F1 > 0.88!
