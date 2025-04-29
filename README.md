# High Performance Computing (HPC) Tutorials

This repository contains a collection of tutorials, code samples, and projects for learning and experimenting with High Performance Computing (HPC) concepts. The focus is on parallel and distributed computing using technologies such as OpenMP, MPI, and CUDA, with practical examples including vector operations, matrix computations, and transformer models.

## Directory Structure

- `DistributedAttention/`  
  High-performance Transformer model implementations with Baseline (CPU), OpenMP, MPI, and CUDA versions. Includes profiling scripts and documentation.
- `tutorial_2/` to `tutorial_14/`  
  Individual tutorials covering various HPC topics, each with code, datasets, and reports.
- `mpi/`  
  Additional MPI code samples and test files.

## Key Features

- **Baseline Transformer**: Naive CPU implementation for reference.
- **OpenMP Optimization**: Multi-threaded parallelism for matrix and attention operations.
- **MPI Optimization**: Distributed computation across multiple nodes.
- **CUDA Acceleration**: GPU-accelerated transformer operations.
- **Profiling & Analysis**: Scripts for LIKWID, gprof, gcov, and visualization tools.

## Getting Started

### Prerequisites
- C++ Compiler (GCC/Clang)
- Python 3.x
- OpenMP
- MPI
- CUDA (for GPU acceleration)

### Example: Build and Run DistributedAttention Baseline
```sh
cd DistributedAttention/baseline
# Compile
g++ -fopenmp -o transformer transformer.cpp
# Run
./transformer
```

### Profiling Example
```sh
# Run profiling script (see DistributedAttention/baseline/script.sh for details)
./script.sh
```

## Profiling & Performance Analysis
- Use LIKWID, gprof, and gcov for performance measurement.
- Visualize profiling data with gprof2dot and Graphviz.

## License
This project is licensed under the MIT License. See `DistributedAttention/LICENSE` for details.

## Contributions
Contributions are welcome! Please fork the repository, raise issues, or submit pull requests.
