// src/cuda_logreg/kernel.cu
#include <cuda_runtime.h>
#include <math.h>

// Sigmoid helper on device
__device__ float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// Forward kernel: pred[i] = sigmoid( X_i · w )
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

// Kernel: compute error[i] = pred[i] - y[i]
__global__ void compute_error(
    const float* pred,   // [n_samples]
    const float* y,      // [n_samples]
    float* error,        // [n_samples]
    int n_samples
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;
    error[i] = pred[i] - y[i];
}

// Kernel: gradient over features
// Each block computes one feature j: grad[j] = (1/n) * sum_i X[i,j] * error[i]
__global__ void compute_gradient(
    const float* X,        // [n_samples, n_features]
    const float* error,    // [n_samples]
    float* grad,           // [n_features]
    int n_samples,
    int n_features
) {
    int j = blockIdx.x; // feature index
    if (j >= n_features) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n_samples; i += blockDim.x) {
        float x_ij = X[i * n_features + j];
        sum += x_ij * error[i];
    }

    // Reduce within block
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    shared_sum[tid] = sum;
    __syncthreads();

    // Simple reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad[j] = shared_sum[0] / static_cast<float>(n_samples);
    }
}

// Kernel: weight update w[j] = w[j] - lr * grad[j]
__global__ void sgd_update(
    float* w,           // [n_features]
    const float* grad,  // [n_features]
    float lr,
    int n_features
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_features) return;
    w[j] -= lr * grad[j];
}

// Launchers callable from host.cu
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

extern "C" void launch_compute_error(
    const float* d_pred,
    const float* d_y,
    float* d_error,
    int n_samples
) {
    int blockSize = 256;
    int numBlocks = (n_samples + blockSize - 1) / blockSize;
    compute_error<<<numBlocks, blockSize>>>(
        d_pred, d_y, d_error, n_samples
    );
}

extern "C" void launch_compute_gradient(
    const float* d_X,
    const float* d_error,
    float* d_grad,
    int n_samples,
    int n_features
) {
    int blockSize = 256;
    int numBlocks = n_features;  // one block per feature
    if (blockSize > 256) blockSize = 256;
    compute_gradient<<<numBlocks, blockSize>>>(
        d_X, d_error, d_grad, n_samples, n_features
    );
}

extern "C" void launch_sgd_update(
    float* d_w,
    const float* d_grad,
    float lr,
    int n_features
) {
    int blockSize = 256;
    int numBlocks = (n_features + blockSize - 1) / blockSize;
    sgd_update<<<numBlocks, blockSize>>>(
        d_w, d_grad, lr, n_features
    );
}
