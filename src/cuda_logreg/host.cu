// src/cuda_logreg/host.cu
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
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

// Simple CSV loader: each row is a sample, comma-separated floats
bool load_csv_matrix(const std::string& path, std::vector<float>& data,
                     int& n_rows, int& n_cols) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        return false;
    }

    std::string line;
    std::vector<std::vector<float>> rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(static_cast<float>(std::stof(cell)));
            }
        }
        if (!row.empty()) {
            rows.push_back(std::move(row));
        }
    }
    file.close();

    if (rows.empty()) {
        std::cerr << "No data read from " << path << "\n";
        return false;
    }

    n_rows = static_cast<int>(rows.size());
    n_cols = static_cast<int>(rows[0].size());
    data.resize(n_rows * n_cols);

    for (int i = 0; i < n_rows; ++i) {
        if (static_cast<int>(rows[i].size()) != n_cols) {
            std::cerr << "Inconsistent column count in " << path << "\n";
            return false;
        }
        for (int j = 0; j < n_cols; ++j) {
            data[i * n_cols + j] = rows[i][j];
        }
    }
    return true;
}

// CSV loader for label vector (one float per line)
bool load_csv_vector(const std::string& path, std::vector<float>& data, int& n_rows) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        return false;
    }
    std::string line;
    data.clear();
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        data.push_back(static_cast<float>(std::stof(line)));
    }
    file.close();
    n_rows = static_cast<int>(data.size());
    return n_rows > 0;
}

int main() {
    std::cout << "CUDA logistic regression – training on heart_disease_uci (train split)\n";

    // 1. Load X_train and y_train from CSV exported by Python
    std::vector<float> h_X, h_y;
    int n_samples = 0, n_features = 0, n_labels = 0;

    if (!load_csv_matrix("data/cuda/X_train_cuda.csv", h_X, n_samples, n_features)) {
        return 1;
    }
    if (!load_csv_vector("data/cuda/y_train_cuda.csv", h_y, n_labels)) {
        return 1;
    }
    if (n_samples != n_labels) {
        std::cerr << "Mismatch: samples=" << n_samples << " labels=" << n_labels << "\n";
        return 1;
    }

    std::cout << "Loaded X_train (" << n_samples << " x " << n_features << ")\n";

    // 2. Allocate host vectors for weights, predictions, error, gradient
    std::vector<float> h_w(n_features, 0.0f);
    std::vector<float> h_pred(n_samples, 0.0f);

    // 3. Allocate device memory
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

    // 4. Train with simple batch gradient descent
    const float lr = 0.1f;
    const int n_iters = 200;

    for (int it = 0; it < n_iters; ++it) {
        launch_logistic_forward(d_X, d_w, d_pred, n_samples, n_features);
        launch_compute_error(d_pred, d_y, d_error, n_samples);
        launch_compute_gradient(d_X, d_error, d_grad, n_samples, n_features);
        launch_sgd_update(d_w, d_grad, lr, n_features);
    }

    // 5. Copy back final weights and predictions
    cudaMemcpy(h_w.data(), d_w, n_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pred.data(), d_pred, n_samples * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_pred);
    cudaFree(d_error);
    cudaFree(d_grad);

    // 6. Compute simple training accuracy on host
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        int y_true = (h_y[i] >= 0.5f) ? 1 : 0;
        int y_hat = (h_pred[i] >= 0.5f) ? 1 : 0;
        if (y_true == y_hat) correct++;
    }
    float train_acc = static_cast<float>(correct) / static_cast<float>(n_samples);

    std::cout << "Training accuracy (CUDA LR on train split): " << train_acc << "\n";
    std::cout << "Done.\n";

    return 0;
}
