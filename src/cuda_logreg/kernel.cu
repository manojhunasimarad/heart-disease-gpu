// src/cuda_logreg/kernel.cu
#include <cuda_runtime.h>
#include <math.h>

// Sigmoid helper on device
__device__ float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// Kernel: for each sample i, compute z_i = X_i · w, then pred_i = sigmoid(z_i)
__global__ void logistic_forward(
    const float* X,      // [n_samples, n_features]
    const float* w,      // [n_features]
    float* pred,         // [n_samples]
    int n_samples,
    int n_features
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // sample index
    if (i >= n_samples) return;

    float z = 0.0f;
    const float* X_row = X + i * n_features;

    for (int j = 0; j < n_features; ++j) {
        z += X_row[j] * w[j];
    }

    pred[i] = sigmoid(z);
}

// C-style launcher for host code
extern "C" void launch_logistic_forward(
    const float* d_X,
    const float* d_w,
    float* d_pred,
    int n_samples,
    int n_features
) {
    int blockSize = 256;
    int numBlocks = (n_samples + blockSize - 1) / blockSize;
    logistic_forward<<<numBlocks, blockSize>>>(
        d_X, d_w, d_pred, n_samples, n_features
    );
}
