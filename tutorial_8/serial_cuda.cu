#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000000
#define THREADS_PER_BLOCK 512

__global__ void sum_n(double *numbers, double *result, int n)
{
    __shared__ double sharedSum[THREADS_PER_BLOCK]; 
    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    sharedSum[tid] = (i < n) ? numbers[i] : 0.0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(result, sharedSum[0]);
    }
}

int main()
{
    FILE *f = fopen("data.txt", "r");
    if (!f)
    {
        printf("Failed to open file.\n");
        return 1;
    }

    double *h_numbers = (double *)malloc(N * sizeof(double));
    double *h_final_sum = (double *)malloc(sizeof(double));
    *h_final_sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        fscanf(f, "%lf", &h_numbers[i]);
    }
    fclose(f);

    double *dev_numbers, *dev_final_sum;
    cudaMalloc(&dev_numbers, N * sizeof(double));
    cudaMemcpy(dev_numbers, h_numbers, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_final_sum, sizeof(double));
    cudaMemset(dev_final_sum, 0, sizeof(double));

    cudaDeviceSynchronize();
    clock_t start = clock();

    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sum_n<<<numBlocks, THREADS_PER_BLOCK>>>(dev_numbers, dev_final_sum, N);

    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(h_final_sum, dev_final_sum, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Sum: %lf\n", *h_final_sum);
    printf("Time taken: %lf seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    cudaFree(dev_numbers);
    cudaFree(dev_final_sum);
    free(h_numbers);
    free(h_final_sum);

    return 0;
}
