#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define SIZE 10000000

#define THREADS_PER_BLOCK 512

#define BLOCKS_PER_GRID ((SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

__global__ void DotProductKernel(double *vector1, double *vector2, double *result_sum) {
    __shared__ double temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < SIZE) {
        temp[threadIdx.x] = vector1[index] * vector2[index];
    } else {
        temp[threadIdx.x] = 0.0;
    }

    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result_sum, temp[0]);
    }
}

int main() {
    FILE *file1 = fopen("dataset1.txt", "r");
    FILE *file2 = fopen("dataset2.txt", "r");
    if (!file1 || !file2) {
        printf("Failed to open file.\n");
        return 1;
    }

    double *host_vector1 = (double *)malloc(SIZE * sizeof(double));
    double *host_vector2 = (double *)malloc(SIZE * sizeof(double));
    double *host_result_sum = (double *)malloc(sizeof(double));
    *host_result_sum = 0.0;

    for (int i = 0; i < SIZE; i++) {
        fscanf(file1, "%lf", &host_vector1[i]);
        fscanf(file2, "%lf", &host_vector2[i]);
    }

    fclose(file1);
    fclose(file2);

    double *device_vector1, *device_vector2, *device_result_sum;
    cudaMalloc(&device_vector1, SIZE * sizeof(double));
    cudaMalloc(&device_vector2, SIZE * sizeof(double));
    cudaMalloc(&device_result_sum, sizeof(double));

    cudaMemcpy(device_vector1, host_vector1, SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector2, host_vector2, SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(device_result_sum, 0, sizeof(double));

    cudaDeviceSynchronize();
    clock_t start_time, end_time;
    start_time = clock();
    DotProductKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_vector1, device_vector2, device_result_sum);
    cudaDeviceSynchronize();
    end_time = clock();

    cudaMemcpy(host_result_sum, device_result_sum, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Dot product: %lf\n", *host_result_sum);
    printf("Time taken: %lf seconds\n", ((double)(end_time - start_time)) / CLOCKS_PER_SEC);

    cudaFree(device_vector1);
    cudaFree(device_vector2);
    cudaFree(device_result_sum);
    free(host_vector1);
    free(host_vector2);
    free(host_result_sum);

    return 0;
}