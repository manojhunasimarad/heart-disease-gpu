# GPU-Accelerated Heart Disease Prediction

Accelerating Logistic Regression kernels using CUDA.

## Setup
1. Build Docker: docker build -t heart-gpu .
2. Run: docker run -it --gpus all -p 8888:8888 heart-gpu

Status: Baseline notebooks + RAPIDS experiments added.