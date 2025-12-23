// src/cuda_logreg/host.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Launcher from kernel.cu
extern "C" void launch_logistic_forward(
    const float* d_X,
    const float* d_w,
    float* d_pred,
    int n_samples,
    int n_features
);

int main() {
    std::cout << "CUDA logistic regression forward pass test\n";

    const int n_samples = 3;
    const int n_features = 2;

    // Tiny example:
    // X = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    // w = [1, -1]
    std::vector<float> h_X = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    std::vector<float> h_w = {1.0f, -1.0f};
    std::vector<float> h_pred(n_samples, 0.0f);

    float *d_X = nullptr, *d_w = nullptr, *d_pred = nullptr;
    cudaMalloc(&d_X, n_samples * n_features * sizeof(float));
    cudaMalloc(&d_w, n_features * sizeof(float));
    cudaMalloc(&d_pred, n_samples * sizeof(float));

    cudaMemcpy(d_X, h_X.data(), n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), n_features * sizeof(float), cudaMemcpyHostToDevice);

    launch_logistic_forward(d_X, d_w, d_pred, n_samples, n_features);

    cudaMemcpy(h_pred.data(), d_pred, n_samples * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_w);
    cudaFree(d_pred);

    std::cout << "Predicted probabilities:\n";
    for (int i = 0; i < n_samples; ++i) {
        std::cout << "sample " << i << " -> " << h_pred[i] << "\n";
    }

    return 0;
}
