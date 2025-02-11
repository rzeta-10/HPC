#!/bin/bash

# Compile the C++ program
g++ -fopenmp matrix_multiplication.cpp -o matrix_multiplication

# Define thread counts
THREADS_LIST="1 2 4 6 8 10 12 16 20 32 64 128 256"

# Remove previous results
rm -f execution_times.txt

echo "Running experiment..."
for threads in $THREADS_LIST; do
    echo "Running with $threads threads..."
    ./matrix_multiplication $threads
done

echo "Experiment completed! Results saved in execution_times.txt"
