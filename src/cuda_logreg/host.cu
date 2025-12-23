// src/cuda_logreg/host.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Launchers from kernel.cu
extern "C" void launch_logistic_forward(
    const float* d_X,
    const float* d_w,
    float* d_pred,
    int n_samples,
    int n_features
);

extern "C" void launch_compute_error(
    const float* d_pred,
    const float* d_y,
    float* d_error,
    int n_samples
);

extern "C" void launch_compute_gradient(
    const float* d_X,
    const float* d_error,
    float* d_grad,
    int n_samples,
    int n_features
);

extern "C" void launch_sgd_update(
    float* d_w,
    const float* d_grad,
    float lr,
    int n_features
);

int main() {
    std::cout << "CUDA logistic regression – toy gradient descent\n";

    const int n_samples = 4;
    const int n_features = 2;
    const float lr = 0.1f;
    const int n_iters = 50;

    // Tiny linearly separable dataset
    // X = [[0,0], [0,1], [1,0], [1,1]]
    // y = [0, 0, 0, 1]  (only last one is positive)
    std::vector<float> h_X = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    std::vector<float> h_y = {0.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> h_w(n_features, 0.0f);          // weights
    std::vector<float> h_pred(n_samples, 0.0f);
    std::vector<float> h_error(n_samples, 0.0f);
    std::vector<float> h_grad(n_features, 0.0f);

    float *d_X = nullptr, *d_y = nullptr;
    float *d_w = nullptr, *d_pred = nullptr;
    float *d_error = nullptr, *d_grad = nullptr;

    cudaMalloc(&d_X, n_samples * n_features * sizeof(float));
    cudaMalloc(&d_y, n_samples * sizeof(float));
    cudaMalloc(&d_w, n_features * sizeof(float));
    cudaMalloc(&d_pred, n_samples * sizeof(float));
    cudaMalloc(&d_error, n_samples * sizeof(float));
    cudaMalloc(&d_grad, n_features * sizeof(float));

    cudaMemcpy(d_X, h_X.data(), n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), n_features * sizeof(float), cudaMemcpyHostToDevice);

    for (int it = 0; it < n_iters; ++it) {
        // forward: pred = sigmoid(Xw)
        launch_logistic_forward(d_X, d_w, d_pred, n_samples, n_features);

        // error = pred - y
        launch_compute_error(d_pred, d_y, d_error, n_samples);

        // grad = X^T (pred - y) / n
        launch_compute_gradient(d_X, d_error, d_grad, n_samples, n_features);

        // w = w - lr * grad
        launch_sgd_update(d_w, d_grad, lr, n_features);
    }

    // Copy back final weights and predictions
    cudaMemcpy(h_w.data(), d_w, n_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pred.data(), d_pred, n_samples * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_pred);
    cudaFree(d_error);
    cudaFree(d_grad);

    std::cout << "Final weights:\n";
    for (int j = 0; j < n_features; ++j) {
        std::cout << "w[" << j << "] = " << h_w[j] << "\n";
    }

    std::cout << "Final predictions:\n";
    for (int i = 0; i < n_samples; ++i) {
        std::cout << "sample " << i << " -> " << h_pred[i] << "\n";
    }

    return 0;
}
