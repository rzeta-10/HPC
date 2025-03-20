#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000

double *host_matrixA = (double *)malloc(N * N * sizeof(double));
double *host_matrixB = (double *)malloc(N * N * sizeof(double));
double *host_result = (double *)malloc(N * N * sizeof(double));

__global__ void matrixMult(double *device_matrixA, double *device_matrixB, double *device_result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int index = row * N + col;
        device_result[index] = device_matrixA[index] * device_matrixB[index];
    }
}

void readMatrixFromFile(const char *filename, double *matrix) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(file, "%lf", &matrix[i * N + j]) != 1) {
                perror("Error reading file");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

int main() {
    if (!host_matrixA || !host_matrixB || !host_result) {
        perror("Error allocating memory");
        exit(EXIT_FAILURE);
    }

    readMatrixFromFile("matrix1.txt", host_matrixA);
    readMatrixFromFile("matrix2.txt", host_matrixB);

    double *device_matrixA, *device_matrixB, *device_result;
    cudaMalloc((void **)&device_matrixA, N * N * sizeof(double));
    cudaMalloc((void **)&device_matrixB, N * N * sizeof(double));
    cudaMalloc((void **)&device_result, N * N * sizeof(double));

    cudaMemcpy(device_matrixA, host_matrixA, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrixB, host_matrixB, N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(device_matrixA, device_matrixB, device_result);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("CUDA Execution Time: %f seconds\n", milliseconds / 1000);

    cudaMemcpy(host_result, device_result, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_matrixA);
    cudaFree(device_matrixB);
    cudaFree(device_result);

    free(host_matrixA);
    free(host_matrixB);
    free(host_result);

    return 0;
}
