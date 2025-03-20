#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 8000000
#define THREADS_PER_BLOCK 1024

__global__ void vector_mul(double *a, double *b, double *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        c[index] = a[index] * b[index];
    }
}

int main(){
    FILE *input1 = fopen("file1.txt", "r");
    FILE *input2 = fopen("file2.txt", "r");

    double *a = (double*)malloc(N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *c_mul = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++){
        fscanf(input1, "%lf", &a[i]);
        fscanf(input2, "%lf", &b[i]);
    }

    fclose(input1);
    fclose(input2);

    double *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N * sizeof(double));
    cudaMalloc((void **)&d_b, N * sizeof(double));
    cudaMalloc((void **)&d_c, N * sizeof(double));

    cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEventRecord(start);
    vector_mul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(c_mul, d_c, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Time taken by parallel multiplication code: %f ms\n", milliseconds);

    // Print the first 10 values of the result
    printf("First 10 values of the result:\n");
    for (int i = 0; i < 10; i++){
        printf("%f\n", c_mul[i]);
    }
    
    free(a);
    free(b);
    free(c_mul);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}