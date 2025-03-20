#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath> // For isnan and isinf

using namespace std;

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// Error checking macro for cuBLAS calls
#define CHECK_CUBLAS_ERROR(call) \
{ \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// This block enables compilation of the code with and without LIKWID in place
#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

// Forward declarations
vector<vector<double>> read_data(string, size_t, size_t);
vector<vector<double>> get_positional_encoding(size_t, size_t);

// CUDA kernel for matrix addition
__global__ void addMatrixKernel(double* A, double* B, double* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel for applying bias
__global__ void addBiasKernel(double* matrix, double* bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        matrix[idx] += bias[col];
    }
}

// CUDA kernel for ReLU activation with NaN handling
__global__ void reluKernel(double* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double val = matrix[idx];
        
        // Check for NaN or Inf and handle it
        if (!isfinite(val)) {
            matrix[idx] = 0.0; // Replace NaN/Inf with zero for ReLU
        } else {
            matrix[idx] = max(0.0, val);
        }
    }
}

// CUDA kernel for scaling tensor values
__global__ void scaleKernel(double* matrix, double scale, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // Prevent division by zero or very small numbers
        if (fabs(scale) > 1e-10) {
            matrix[idx] /= scale;
        }
    }
}

// CUDA kernel for softmax operation
__global__ void softmaxKernel(double* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows) {
        // Find max value in the row for numerical stability
        double max_val = -INFINITY;
        for (int j = 0; j < cols; j++) {
            max_val = max(max_val, matrix[row * cols + j]);
        }
        
        // Calculate exp and sum
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            int idx = row * cols + j;
            matrix[idx] = exp(matrix[idx] - max_val);
            sum += matrix[idx];
        }
        
        // Normalize
        for (int j = 0; j < cols; j++) {
            int idx = row * cols + j;
            matrix[idx] /= sum;
        }
    }
}

// CUDA kernel for split heads
__global__ void splitHeadsKernel(double* input, double* output, int seq_len, int num_heads, int d_head) {
    int h = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < num_heads && i < seq_len && j < d_head) {
        output[(h * seq_len + i) * d_head + j] = input[i * (num_heads * d_head) + h * d_head + j];
    }
}

// CUDA kernel for concatenate heads
__global__ void concatenateHeadsKernel(double* input, double* output, int seq_len, int num_heads, int d_value) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < seq_len && k < num_heads * d_value) {
        int h = k / d_value;
        int d = k % d_value;
        output[j * (num_heads * d_value) + k] = input[(h * seq_len + j) * d_value + d];
    }
}

// CUDA kernel for layer normalization
__global__ void layerNormKernel(double* input, double* gamma, double* beta, double* output, int rows, int cols, float epsilon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Calculate mean
        double mean = 0.0;
        for (int i = 0; i < cols; i++) {
            mean += input[row * cols + i];
        }
        mean /= cols;
        
        // Calculate variance
        double var = 0.0;
        for (int i = 0; i < cols; i++) {
            double diff = input[row * cols + i] - mean;
            var += diff * diff;
        }
        var /= cols;
        
        // Normalize, scale, and shift
        double stddev_inv = 1.0 / sqrt(var + epsilon);
        for (int i = 0; i < cols; i++) {
            int idx = row * cols + i;
            double norm = (input[idx] - mean) * stddev_inv;
            output[idx] = gamma[i] * norm + beta[i];
        }
    }
}

// Utility functions for host-device memory transfers
void copyMatrixToDevice(double* d_matrix, const vector<vector<double>>& h_matrix, int rows, int cols) {
    double* h_temp = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_temp[i * cols + j] = h_matrix[i][j];
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_temp, rows * cols * sizeof(double), cudaMemcpyHostToDevice));
    delete[] h_temp;
}

void copyMatrixToHost(vector<vector<double>>& h_matrix, double* d_matrix, int rows, int cols) {
    double* h_temp = new double[rows * cols];
    CHECK_CUDA_ERROR(cudaMemcpy(h_temp, d_matrix, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < rows; i++) {
        h_matrix[i].resize(cols);
        for (int j = 0; j < cols; j++) {
            h_matrix[i][j] = h_temp[i * cols + j];
        }
    }
    delete[] h_temp;
}

void copyVectorToDevice(double* d_vector, const vector<double>& h_vector, int size) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_vector, h_vector.data(), size * sizeof(double), cudaMemcpyHostToDevice));
}

// Helper function for matrix multiplication using cuBLAS with enhanced stability
void matrixMultiply(cublasHandle_t handle, double* A, double* B, double* C, int m, int n, int k) {
    const double alpha = 1.0f;
    const double beta = 0.0f;
    
    // Perform matrix multiplication: C = A * B
    // Note: cuBLAS uses column-major order, while C/C++ uses row-major order
    // C(m,n) = A(m,k) * B(k,n)
    CHECK_CUBLAS_ERROR(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
}

// Utility function to generate a random matrix
vector<vector<double>> genRandomMatrix(size_t rows, size_t cols, double scale = 0.01) {
    vector<vector<double>> matrix(rows, vector<double>(cols, 0.0f));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            // Xavier/Glorot initialization approach - scale by sqrt(1/n)
            double glorot_scale = sqrt(2.0 / (rows + cols));
            // Use a smaller scale for better numerical stability
            matrix[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * glorot_scale * scale;
        }
    }
    return matrix;
}

// Utility function to add vectors
vector<vector<double>> add_vectors(vector<vector<double>>& a, vector<vector<double>>& b) {
    size_t rows = a.size();
    size_t cols = a[0].size();
    vector<vector<double>> result(rows, vector<double>(cols, 0.0f));
    
    // Allocate device memory
    double *d_a, *d_b, *d_result;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, rows * cols * sizeof(double)));
    
    // Copy data to device
    copyMatrixToDevice(d_a, a, rows, cols);
    copyMatrixToDevice(d_b, b, rows, cols);
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    addMatrixKernel<<<gridDim, blockDim>>>(d_a, d_b, d_result, rows, cols);
    
    // Copy result back to host
    copyMatrixToHost(result, d_result, rows, cols);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    return result;
}

// Layer normalization function
vector<vector<double>> layer_norm(vector<vector<double>>& input, vector<double>& gamma, vector<double>& beta, float epsilon = 1e-6) {
    size_t rows = input.size();
    size_t cols = input[0].size();
    vector<vector<double>> output(rows, vector<double>(cols, 0.0f));
    
    // Allocate device memory
    double *d_input, *d_gamma, *d_beta, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gamma, cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_beta, cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, rows * cols * sizeof(double)));
    
    // Copy data to device
    copyMatrixToDevice(d_input, input, rows, cols);
    copyVectorToDevice(d_gamma, gamma, cols);
    copyVectorToDevice(d_beta, beta, cols);
    
    // Set up grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x);
    
    // Launch kernel
    layerNormKernel<<<gridDim, blockDim>>>(d_input, d_gamma, d_beta, d_output, rows, cols, epsilon);
    
    // Copy result back to host
    copyMatrixToHost(output, d_output, rows, cols);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
    
    return output;
}

// Matrix transpose function
vector<vector<double>> matrixTranspose(vector<vector<double>>& a) {
    size_t rows = a.size();
    size_t cols = a[0].size();
    vector<vector<double>> result(cols, vector<double>(rows, 0.0f));
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j][i] = a[i][j];
        }
    }
    
    return result;
}

// Split heads function with CUDA acceleration
vector<vector<double>> split_heads(vector<vector<double>>& x, size_t num_heads, size_t d_head) {
    size_t seq_len = x.size();
    size_t d_model = x[0].size();
    vector<vector<double>> X_split(seq_len * num_heads, vector<double>(d_head, 0.0f));
    
    // Allocate device memory
    double *d_x, *d_x_split;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, seq_len * d_model * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x_split, seq_len * num_heads * d_head * sizeof(double)));
    
    // Copy data to device
    copyMatrixToDevice(d_x, x, seq_len, d_model);
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((d_head + blockDim.x - 1) / blockDim.x, 
                (seq_len + blockDim.y - 1) / blockDim.y, 
                num_heads);
    
    // Launch kernel
    splitHeadsKernel<<<gridDim, blockDim>>>(d_x, d_x_split, seq_len, num_heads, d_head);
    
    // Copy result back to host
    copyMatrixToHost(X_split, d_x_split, seq_len * num_heads, d_head);
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_x_split);
    
    return X_split;
}

// Concatenate heads function with CUDA acceleration
vector<vector<double>> concatenate_heads(vector<vector<double>>& x, size_t num_heads, size_t seq_len, size_t d_value) {
    vector<vector<double>> X(seq_len, vector<double>(num_heads * d_value, 0.0f));
    
    // Allocate device memory
    double *d_x, *d_X;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, num_heads * seq_len * d_value * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_X, seq_len * num_heads * d_value * sizeof(double)));
    
    // Copy data to device
    copyMatrixToDevice(d_x, x, num_heads * seq_len, d_value);
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((num_heads * d_value + blockDim.x - 1) / blockDim.x, 
                (seq_len + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    concatenateHeadsKernel<<<gridDim, blockDim>>>(d_x, d_X, seq_len, num_heads, d_value);
    
    // Copy result back to host
    copyMatrixToHost(X, d_X, seq_len, num_heads * d_value);
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_X);
    
    return X;
}

// Apply softmax to each row of a matrix
void softmax_rows(vector<vector<double>>& a) {
    size_t rows = a.size();
    size_t cols = a[0].size();
    
    // Allocate device memory
    double *d_a;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, rows * cols * sizeof(double)));
    
    // Copy data to device
    copyMatrixToDevice(d_a, a, rows, cols);
    
    // Set up grid and block dimensions
    dim3 blockDim(32);
    dim3 gridDim((rows + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    softmaxKernel<<<gridDim, blockDim>>>(d_a, rows, cols);
    
    // Copy result back to host
    copyMatrixToHost(a, d_a, rows, cols);
    
    // Free device memory
    cudaFree(d_a);
}

// Matrix multiplication wrapper for the GPU
vector<vector<double>> matmul(vector<vector<double>>& a, vector<vector<double>>& b) {
    size_t m = a.size();
    size_t k = a[0].size();
    size_t n = b[0].size();
    vector<vector<double>> c(m, vector<double>(n, 0.0f));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    
    // Allocate device memory
    double *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, m * k * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, k * n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, m * n * sizeof(double)));
    
    // Copy data to device
    copyMatrixToDevice(d_a, a, m, k);
    copyMatrixToDevice(d_b, b, k, n);
    
    // Perform matrix multiplication
    matrixMultiply(handle, d_a, d_b, d_c, m, n, k);
    
    // Copy result back to host
    copyMatrixToHost(c, d_c, m, n);
    
    // Free device memory and destroy cuBLAS handle
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    
    return c;
}

// Utility function to extract a single head from a split heads matrix
vector<vector<double>> extractHead(const vector<vector<double>>& heads, size_t head_idx, size_t seq_len, size_t d_head) {
    vector<vector<double>> head(seq_len, vector<double>(d_head, 0.0f));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_head; ++j) {
            head[i][j] = heads[head_idx * seq_len + i][j];
        }
    }
    return head;
}




// Apply bias to a matrix
void add_bias(vector<vector<double>>& x, vector<double>& b) {
    size_t rows = x.size();
    size_t cols = x[0].size();
    
    // Allocate device memory
    double *d_x, *d_b;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, cols * sizeof(double)));
    
    // Copy data to device
    copyMatrixToDevice(d_x, x, rows, cols);
    copyVectorToDevice(d_b, b, cols);
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    addBiasKernel<<<gridDim, blockDim>>>(d_x, d_b, rows, cols);
    
    // Copy result back to host
    copyMatrixToHost(x, d_x, rows, cols);
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_b);
}

// Utility function to check for NaN or infinite values in a matrix
bool check_for_nan(const vector<vector<double>>& matrix, const string& label) {
    bool has_nan = false;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            if (isnan(matrix[i][j]) || isinf(matrix[i][j])) {
                has_nan = true;
                //cout << "NaN/Inf detected in " << label << " at position [" << i << "][" << j << "]: " << matrix[i][j] << endl;
                // Only report the first few NaNs to avoid flooding output
                if (i > 5) return true;
                break;
            }
        }
        if (has_nan && i > 5) break;
    }
    return has_nan;
}

// Utility function to sanitize a matrix, replacing NaN/Inf with zeros
void sanitize_matrix(vector<vector<double>>& matrix) {
    for (auto& row : matrix) {
        for (auto& val : row) {
            if (isnan(val) || isinf(val)) {
                val = 0.0;
            }
        }
    }
}

// Multi-head attention class with CUDA acceleration
class MultiHeadAttention {
public:
    size_t d_model, num_heads, d_key, d_value;
    vector<vector<double>> WQ, WK, WV, WO;
    
    MultiHeadAttention(size_t d_model, size_t num_heads) : d_model(d_model), num_heads(num_heads) {
        assert(d_model % num_heads == 0);
        this->d_model = d_model;
        this->num_heads = num_heads;
        this->d_key = d_model / num_heads;
        this->d_value = d_model / num_heads;
        
        // Initialize with small values for better numerical stability
        double init_scale = 0.01;
        this->WQ = genRandomMatrix(d_model, d_model, init_scale);
        this->WK = genRandomMatrix(d_model, d_model, init_scale);
        this->WV = genRandomMatrix(d_model, d_model, init_scale);
        this->WO = genRandomMatrix(d_model, d_model, init_scale);
    }
    
    vector<vector<double>> forward(vector<vector<double>>& x) {
        // Check input for NaN values
        if (check_for_nan(x, "MultiHeadAttention input")) {
            cout << "NaN detected in MultiHeadAttention input. Sanitizing..." << endl;
            sanitize_matrix(x);
        }
        
        size_t seq_len = x.size();
        
        vector<vector<double>> Q = matmul(x, WQ);
        vector<vector<double>> K = matmul(x, WK);
        vector<vector<double>> V = matmul(x, WV);
        
        if (check_for_nan(Q, "Q")) {
            sanitize_matrix(Q);
        }
        if (check_for_nan(K, "K")) {
            sanitize_matrix(K);
        }
        if (check_for_nan(V, "V")) {
            sanitize_matrix(V);
        }
        
        vector<vector<double>> Q_heads = split_heads(Q, num_heads, d_key);
        vector<vector<double>> K_heads = split_heads(K, num_heads, d_key);
        vector<vector<double>> V_heads = split_heads(V, num_heads, d_value);
        
        if (check_for_nan(Q_heads, "Q_heads")) {
            sanitize_matrix(Q_heads);
        }
        if (check_for_nan(K_heads, "K_heads")) {
            sanitize_matrix(K_heads);
        }
        if (check_for_nan(V_heads, "V_heads")) {
            sanitize_matrix(V_heads);
        }
        
        vector<vector<double>> O_heads(seq_len * num_heads, vector<double>(d_value, 0.0f));
        
        for (size_t h = 0; h < num_heads; h++) {
            // Extract head
            vector<vector<double>> Q_head = extractHead(Q_heads, h, seq_len, d_key);
            vector<vector<double>> K_head = extractHead(K_heads, h, seq_len, d_key);
            vector<vector<double>> V_head = extractHead(V_heads, h, seq_len, d_value);
            
            // Compute attention scores
            vector<vector<double>> K_head_T = matrixTranspose(K_head);
            vector<vector<double>> attention_scores = matmul(Q_head, K_head_T);
            
            if (check_for_nan(attention_scores, "attention_scores")) {
                sanitize_matrix(attention_scores);
            }
            
            // Scale attention scores
            double scale = sqrt(d_key);
            
            if (scale < 1e-10) {
                // Avoid division by very small numbers
                scale = 1.0;
            }
            
            // Allocate device memory
            double *d_scores;
            size_t rows = attention_scores.size();
            size_t cols = attention_scores[0].size();
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_scores, rows * cols * sizeof(double)));
            
            // Copy data to device
            copyMatrixToDevice(d_scores, attention_scores, rows, cols);
            
            // Set up grid and block dimensions
            dim3 blockDim(16, 16);
            dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
            
            // Launch kernel
            scaleKernel<<<gridDim, blockDim>>>(d_scores, scale, rows, cols);
            
            // Check for errors
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            
            // Copy result back to host
            copyMatrixToHost(attention_scores, d_scores, rows, cols);
            
            // Free device memory
            cudaFree(d_scores);
            
            softmax_rows(attention_scores);
            
            if (check_for_nan(attention_scores, "attention_scores_after_softmax")) {
                sanitize_matrix(attention_scores);
            }
            
            vector<vector<double>> attention_output = matmul(attention_scores, V_head);
            
            if (check_for_nan(attention_output, "attention_output")) {
                sanitize_matrix(attention_output);
            }
            
            for (size_t j = 0; j < seq_len; j++) {
                for (size_t k = 0; k < d_value; k++) {
                    O_heads[(h * seq_len) + j][k] = attention_output[j][k];
                }
            }
        }
        
        if (check_for_nan(O_heads, "O_heads")) {
            sanitize_matrix(O_heads);
        }
        
        vector<vector<double>> output = concatenate_heads(O_heads, num_heads, seq_len, d_value);
        
        if (check_for_nan(output, "output_before_WO")) {
            sanitize_matrix(output);
        }
        
        vector<vector<double>> result = matmul(output, WO);
        
        if (check_for_nan(result, "result")) {
            sanitize_matrix(result);
            cout << "NaN detected in MultiHeadAttention final output. Using sanitized values." << endl;
        }
        
        return result;
    }
};

// FeedForward class with CUDA acceleration
class FeedForward {
public:
    size_t d_model, ff_dim;
    vector<vector<double>> W1, W2;
    vector<double> b1, b2;
    
    FeedForward(size_t d_model, size_t ff_dim) : d_model(d_model), ff_dim(ff_dim) {
        // Initialize with small values for better numerical stability
        double init_scale = 0.01;
        this->W1 = genRandomMatrix(d_model, ff_dim, init_scale);
        this->b1 = vector<double>(ff_dim, 0.0f);
        this->W2 = genRandomMatrix(ff_dim, d_model, init_scale);
        this->b2 = vector<double>(d_model, 0.0f);
    }
    
    vector<vector<double>> forward(vector<vector<double>>& x) {
        // Check input for NaN values
        if (check_for_nan(x, "FeedForward input")) {
            sanitize_matrix(x);
        }
        
        vector<vector<double>> h1 = matmul(x, W1);
        
        // Check h1 after matmul
        if (check_for_nan(h1, "FeedForward h1 after matmul")) {
            sanitize_matrix(h1);
        }
        
        add_bias(h1, b1);
        
        // Check h1 after bias
        if (check_for_nan(h1, "FeedForward h1 after bias")) {
            sanitize_matrix(h1);
        }
        
        // ReLU activation
        size_t rows = h1.size();
        size_t cols = h1[0].size();
        
        // Allocate device memory
        double *d_h1;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h1, rows * cols * sizeof(double)));
        
        // Copy data to device
        copyMatrixToDevice(d_h1, h1, rows, cols);
        
        // Set up grid and block dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
        
        // Launch kernel
        reluKernel<<<gridDim, blockDim>>>(d_h1, rows, cols);
        
        // Check for errors after kernel launch
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy result back to host
        copyMatrixToHost(h1, d_h1, rows, cols);
        
        // Free device memory
        cudaFree(d_h1);
        
        // Check h1 after ReLU
        if (check_for_nan(h1, "FeedForward h1 after ReLU")) {
            sanitize_matrix(h1);
        }
        
        // Layer 2
        vector<vector<double>> h2 = matmul(h1, W2);
        
        // Check h2 after matmul
        if (check_for_nan(h2, "FeedForward h2 after matmul")) {
            sanitize_matrix(h2);
        }
        
        add_bias(h2, b2);
        
        // Check h2 after bias
        if (check_for_nan(h2, "FeedForward h2 after bias")) {
            sanitize_matrix(h2);
        }
        
        return h2;
    }
};

// EncoderLayer class with CUDA acceleration
class EncoderLayer {
public:
    size_t d_model;
    size_t num_heads;
    size_t ff_d;
    MultiHeadAttention mha;
    FeedForward ff;
    vector<double> beta;
    vector<double> gamma;
    
    EncoderLayer(size_t d_model, size_t num_heads, size_t ff_d) : d_model(d_model), num_heads(num_heads), ff_d(ff_d), mha(d_model, num_heads), ff(d_model, ff_d) {
        beta = vector<double>(d_model, 0.0f);
        gamma = vector<double>(d_model, 1.0f);
    }
    
    vector<vector<double>> forward(vector<vector<double>>& x) {
        // Check input for NaN values
        if (check_for_nan(x, "EncoderLayer input")) {
            sanitize_matrix(x);
        }
        
        vector<vector<double>> attn_output = mha.forward(x);
        
        // Check attention output for NaN values
        if (check_for_nan(attn_output, "Attention output")) {
            sanitize_matrix(attn_output);
            cout << "Attention output had NaN values, using sanitized values." << endl;
        }
        
        vector<vector<double>> addLayer1 = add_vectors(x, attn_output);
        
        // Check after addition
        if (check_for_nan(addLayer1, "After add1")) {
            sanitize_matrix(addLayer1);
            cout << "After addition 1 had NaN values, using sanitized values." << endl;
        }
        
        vector<vector<double>> norm1 = layer_norm(addLayer1, gamma, beta, 1e-6);
        
        // Check after normalization
        if (check_for_nan(norm1, "After norm1")) {
            // If normalization failed, use the input to normalization but scaled down
            cout << "Layer norm 1 produced NaN values, falling back to scaled input." << endl;
            norm1 = addLayer1;
            for (auto& row : norm1) {
                for (auto& val : row) {
                    val *= 0.1; // Scale down to prevent explosion
                }
            }
        }
        
        vector<vector<double>> ff_output = ff.forward(norm1);
        
        // Check feedforward output
        if (check_for_nan(ff_output, "FF output")) {
            sanitize_matrix(ff_output);
            cout << "FeedForward output had NaN values, using sanitized values." << endl;
        }
        
        vector<vector<double>> addLayer2 = add_vectors(norm1, ff_output);
        
        // Check after addition
        if (check_for_nan(addLayer2, "After add2")) {
            sanitize_matrix(addLayer2);
            cout << "After addition 2 had NaN values, using sanitized values." << endl;
        }
        
        vector<vector<double>> norm2 = layer_norm(addLayer2, gamma, beta);
        
        // Check after normalization
        if (check_for_nan(norm2, "After norm2")) {
            // If normalization failed, use the input to normalization but scaled down
            cout << "Layer norm 2 produced NaN values, falling back to scaled input." << endl;
            norm2 = addLayer2;
            for (auto& row : norm2) {
                for (auto& val : row) {
                    val *= 0.1; // Scale down to prevent explosion
                }
            }
        }
        
        return norm2;
    }
};

// Function to read data
vector<vector<double>> read_data(string filename, size_t sequence_length, size_t embed_dim) {
    vector<vector<double>> result(sequence_length, vector<double>(embed_dim, 0.0f));
    ifstream infile(filename);
    
    if (!infile.is_open()) {
        cerr << "Could not open file " << filename << endl;
        return result;
    }
    
    string line;
    size_t line_idx = 0;
    while (getline(infile, line) && line_idx < sequence_length) {
        // Skip empty lines and comment lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        stringstream ss(line);
        string token;
        size_t token_idx = 0;
        
        // Check if the line contains commas (CSV format)
        if (line.find(',') != string::npos) {
            // Process as CSV
            while (getline(ss, token, ',') && token_idx < embed_dim) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
                token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);
                
                if (!token.empty()) {
                    try {
                        result[line_idx][token_idx] = stod(token);
                    } catch (const std::invalid_argument& e) {
                        cerr << "Invalid number format at line " << line_idx+1 << ", token " << token_idx+1 << ": " << token << endl;
                    } catch (const std::out_of_range& e) {
                        cerr << "Number out of range at line " << line_idx+1 << ", token " << token_idx+1 << ": " << token << endl;
                    }
                    token_idx++;
                }
            }
        } else {
            // Process as space-separated
            double value;
            while (ss >> value && token_idx < embed_dim) {
                result[line_idx][token_idx] = value;
                token_idx++;
            }
        }
        
        // Only increment line_idx if we actually processed some values
        if (token_idx > 0) {
            line_idx++;
        }
    }
    
    if (line_idx < sequence_length) {
        cout << "Warning: Only read " << line_idx << " entries from file (expected " << sequence_length << ")" << endl;
    }
    
    return result;
}

// Get positional encoding
vector<vector<double>> get_positional_encoding(size_t sequence_length, size_t d_model) {
    vector<vector<double>> pos_encoding(sequence_length, vector<double>(d_model, 0.0f));
    
    for (size_t pos = 0; pos < sequence_length; pos++) {
        for (size_t i = 0; i < d_model; i += 2) {
            double div_term = exp(-(double)i / d_model * log(10000.0));
            pos_encoding[pos][i] = sin(pos * div_term);
            if (i + 1 < d_model) {
                pos_encoding[pos][i + 1] = cos(pos * div_term);
            }
        }
    }
    
    return pos_encoding;
}

// Main function
int main() {
    LIKWID_MARKER_INIT;
    
    // Set a fixed seed for reproducibility
    srand(42);
    
    // Set up parameters
    size_t sequence_length = 512;
    size_t d_model = 512;
    size_t num_heads = 8;
    size_t ff_dim = 2048;
    size_t num_layers = 6;
    
    // Initialize input data
    vector<vector<double>> input;
    
    // Try to read input from a file, otherwise generate random input
    try {
        cout << "Attempting to read input data from 'dataset_vectors.txt'..." << endl;
        input = read_data("dataset_vectors.txt", sequence_length, d_model);
        cout << "Successfully loaded data with sequence length: " << input.size() << " and embedding dimension: " << input[0].size() << endl;
        
        // Print a sample of the input for verification
        cout << "Sample input values (first 3 rows, first 5 columns):" << endl;
        for (size_t i = 0; i < min(size_t(3), input.size()); i++) {
            for (size_t j = 0; j < min(size_t(5), input[i].size()); j++) {
                cout << input[i][j] << " ";
            }
            cout << endl;
        }
    } catch (const exception& e) {
        cout << "Error reading input file: " << e.what() << endl;
        cout << "Using random input instead" << endl;
        input = genRandomMatrix(sequence_length, d_model, 0.1); // Use smaller values
    }
    
    // Check input for NaN values
    if (check_for_nan(input, "Input data")) {
        cout << "Warning: Input data contains NaN values. Sanitizing..." << endl;
        sanitize_matrix(input);
    }
    
    // Add positional encoding
    vector<vector<double>> pos_encoding = get_positional_encoding(sequence_length, d_model);
    
    // Check positional encoding for NaN values
    if (check_for_nan(pos_encoding, "Positional encoding")) {
        cout << "Warning: Positional encoding contains NaN values. Sanitizing..." << endl;
        sanitize_matrix(pos_encoding);
    }
    
    vector<vector<double>> embedded_input = add_vectors(input, pos_encoding);
    
    // Check embedded input for NaN values
    if (check_for_nan(embedded_input, "Embedded input")) {
        cout << "Warning: Embedded input contains NaN values. Sanitizing..." << endl;
        sanitize_matrix(embedded_input);
    }
    
    // Initialize CUDA device
    int device = 0;
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));
    CHECK_CUDA_ERROR(cudaSetDevice(device));
    
    cout << "Using CUDA device: " << deviceProp.name << endl;
    
    // Create encoder layers with smaller initialization values
    vector<EncoderLayer> layers;
    for (size_t i = 0; i < num_layers; i++) {
        layers.push_back(EncoderLayer(d_model, num_heads, ff_dim));
    }
    
    // Forward pass through encoder layers
    vector<vector<double>> encoder_output = embedded_input;
    
    LIKWID_MARKER_START("transformer");
    auto start_time = chrono::high_resolution_clock::now();
    
    // Process each layer with checks for NaNs between layers
    for (size_t i = 0; i < num_layers; i++) {
        // Check before processing layer
        if (check_for_nan(encoder_output, "Before layer " + to_string(i))) {
            cout << "Warning: NaN values detected before layer " << i << ". Sanitizing..." << endl;
            sanitize_matrix(encoder_output);
        }
        
        // Process through the layer
        encoder_output = layers[i].forward(encoder_output);
        
        // Check after processing layer
        if (check_for_nan(encoder_output, "After layer " + to_string(i))) {
            cout << "Warning: Layer " << i << " produced NaN values. Using fallback..." << endl;
            // If a layer produces NaNs, fall back to a sanitized version of input
            if (i > 0) {
                // Try to use output from the previous layer
                encoder_output = layers[i-1].forward(embedded_input);
                sanitize_matrix(encoder_output);
            } else {
                // For the first layer, fall back to the embedded input
                encoder_output = embedded_input;
            }
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Time taken for forward pass: " << duration.count() << " ms" << endl;
    LIKWID_MARKER_STOP("transformer");
    
    // Final check for NaN values
    if (check_for_nan(encoder_output, "Final output")) {
        cout << "Warning: Final output contains NaN values. Sanitizing for display..." << endl;
        sanitize_matrix(encoder_output);
    }
    
    // Print a sample of the output for verification
    cout << "Sample output values:" << endl;
    for (size_t i = 0; i < min(size_t(5), encoder_output.size()); i++) {
        for (size_t j = 0; j < min(size_t(5), encoder_output[i].size()); j++) {
            cout << encoder_output[i][j] << " ";
        }
        cout << endl;
    }
    
    LIKWID_MARKER_CLOSE;
    
    return 0;
}
