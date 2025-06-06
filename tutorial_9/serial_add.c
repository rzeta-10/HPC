#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 8000000

void vector_add_serial(double *a, double *b, double *c){
    for (int i = 0; i < N; i++){
        c[i] = a[i] + b[i];
    }
}

int main(){
    FILE *input1 = fopen("file1.txt", "r");
    FILE *input2 = fopen("file2.txt", "r");

    double *a = (double*)malloc(N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *c = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++){
        fscanf(input1, "%lf", &a[i]);
        fscanf(input2, "%lf", &b[i]);
    }

    clock_t start = clock();
    vector_add_serial(a, b, c);
    clock_t end = clock();
    printf("Time taken for serial addition: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Print the first 10 elements of the result
    printf("First 10 elements of the result:\n");
    for (int i = 0; i < 10; i++){
        printf("%lf\n", c[i]);
    }

    // Free allocated memory and close files
    free(a);
    free(b);
    free(c);
    fclose(input1);
    fclose(input2);

    return 0;
}