#!/bin/bash

# Compile the C++ code
g++ -fopenmp matrix_addition.cpp -o matrix_addition

# Remove previous results
rm -f execution_times.txt

# Define thread counts to test
THREADS=(1 2 4 6 8 10 12 16 20 32 64 128 256)

# Run the program for each thread count and store the results
for t in "${THREADS[@]}"; do
    echo "Running with $t threads..."
    ./matrix_addition $t
done

echo "Experiment completed! Results saved in execution_times.txt"
