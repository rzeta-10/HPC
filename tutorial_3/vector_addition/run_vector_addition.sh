#!/bin/bash

# Compile the program
g++ -fopenmp vector_addition.cpp -o vector_addition

# List of thread counts to test
threads=(1 2 4 6 8 10 12 16 20 32 64)
output_file="performance_results.txt"

# Clear previous results
echo "Threads,Time" > $output_file

# Run the program for each thread count
for t in "${threads[@]}"
do
    echo "Running with $t threads..."
    
    # Run the program and capture output
    output=$(./vector_addition $t)

    # Extract the time using grep and awk (second last field)
    time_value=$(echo "$output" | grep "Time:" | awk '{print $(NF-1)}')

    # Save result in CSV file
    echo "$t,$time_value" >> $output_file
done

echo "Results saved to $output_file"
