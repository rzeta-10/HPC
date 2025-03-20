#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define N 8000000
#define THREADS_PER_BLOCK 1024

__global__ void vector_add(double *a, double *b, double *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        c[index] = a[index] + b[index];
    }
}

int main(){
    FILE *input1 = fopen("file1.txt", "r");
    FILE *input2 = fopen("file2.txt", "r");

    double *a = (double*)malloc(N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *c_add = (double*)malloc(N * sizeof(double));

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

    cudaDeviceSynchronize();
    clock_t start = clock();
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(c_add, d_c, N * sizeof(double), cudaMemcpyDeviceToHost);

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken by parallel addition code: %f\n", time_taken);

    // Print the first 10 elements of the result
    printf("First 10 elements of the result:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f\n", c_add[i]);
    }
    
    free(a);
    free(b);
    free(c_add);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}