#!/bin/bash

# Compile the MPI C++ program
mpic++ transformer_mpi.cpp -o transformer_mpi
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting..."
    exit 1
fi

PROCESSES_LIST="1 2 4 6 8 10 12 16 20 32 64 128"

# Remove previous results
rm -f execution_times.txt

echo "Running MPI experiment..."
for procs in $PROCESSES_LIST; do
    echo "Running with $procs processes..."
    mpirun --oversubscribe -np $procs ./transformer_mpi
    if [ $? -ne 0 ]; then
        echo "Execution failed with $procs processes!"
    fi
done

echo "MPI experiment completed! Results saved in execution_times.txt"