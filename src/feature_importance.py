# Add to end of gpu_baseline.py
print("\n📊 TOP FEATURES:")
importances = rf_gpu.feature_importances_
feature_names = [f'feat_{i}' for i in range(len(importances))]
top_features = sorted(zip(importances, feature_names), reverse=True)[:10]
for imp, name in top_features:
    print(f"{name}: {imp:.4f}")
