#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define PI 3.1415926535897932
#define DEFAULT_ITERATIONS 1000000

int main(int argc, char* argv[]) {
    int iterations, i, count = 0, reduced_count = 0, rank, size;
    double x, y, z, calculated_pi;
    time_t start_time, current_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Rank %d running on host: %s\n", rank, hostname);
    fflush(stdout);  // Add flush to ensure output is displayed
    
    // Add barrier to ensure all processes have reported their hostnames
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("All processes have reported their hostnames\n");
        fflush(stdout);
    }
    
    if (argc > 1) {
        iterations = atoi(argv[1]);
    } else {
        iterations = DEFAULT_ITERATIONS;
    }
    
    // Record start time
    start_time = time(NULL);
    
    // Initialize random seed
    srand48(time(NULL) + rank);
    
    // Add progress reporting
    int report_interval = iterations / 10; // Report every 10%
    
    if (rank == 0) {
        printf("Starting calculations with %d iterations per process...\n", iterations);
        fflush(stdout);
    }
    
    for (i = 0; i < iterations; i++) {
        x = drand48();
        y = drand48();
        z = x*x + y*y;
        if (z <= 1) {
            count++;
        }
        
        // Progress reporting
        if (report_interval > 0 && i > 0 && i % report_interval == 0) {
            current_time = time(NULL);
            printf("Rank %d: %d%% complete (%d/%d iterations) - %ld seconds elapsed\n", 
                   rank, (i * 100) / iterations, i, iterations, 
                   (long)(current_time - start_time));
            fflush(stdout);  // Ensure progress is displayed
        }
    }

    // Print completion message
    current_time = time(NULL);
    printf("Rank %d: Calculation complete in %ld seconds\n", 
           rank, (long)(current_time - start_time));
    fflush(stdout);
    
    // Add barrier before reduction to ensure all processes complete
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("All processes have completed calculations, gathering results...\n");
        fflush(stdout);
    }
    
    MPI_Reduce(&count, &reduced_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int total_iterations = iterations * size;
        calculated_pi = ((double)reduced_count / (double)total_iterations) * 4.0;
        printf("\n====== RESULTS ======\n");
        printf("Number of processes: %d\n", size);
        printf("Iterations per process: %d\n", iterations);
        printf("Total iterations: %d\n", total_iterations);
        printf("Points inside circle: %d\n", reduced_count);
        printf("Actual PI: %.15lf\n", PI);
        printf("Calculated PI: %.15lf\n", calculated_pi);
        printf("Difference: %.15lf\n", fabs(PI - calculated_pi));
        printf("Total execution time: %ld seconds\n", (long)(current_time - start_time));
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}