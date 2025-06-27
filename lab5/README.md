# Parallel Matrix Multiplication Lab

This project implements parallel matrix multiplication using Pthreads and a custom parallel_for implementation. The goal is to analyze the performance characteristics of parallel matrix multiplication with different matrix sizes and thread counts.

## Project Structure

```
.
├── src/
│   ├── matrix_mult_pthread.c    # Matrix multiplication using parallel_for
│   └── parallel_for.c           # Custom parallel_for implementation
├── results/
│   ├── plots/                   # Performance analysis plots
│   ├── summary.csv             # Raw performance data
│   └── summary_statistics.csv  # Statistical analysis
├── Makefile                    # Build configuration
├── run_experiments.sh          # Script to run experiments
└── analyze_results.py          # Script to analyze results
```

## Requirements

- GCC compiler
- Python 3.x with pandas, matplotlib, and seaborn packages
- POSIX threads (Pthreads) library

## Building the Project

```bash
make clean
make
```

## Running Experiments

The experiments can be run using:

```bash
./run_experiments.sh
```

This will:
1. Compile the source code
2. Run matrix multiplication with different matrix sizes (128-2048) and thread counts (1-16)
3. Save the results in the `results` directory

## Analyzing Results

To analyze the results and generate plots:

```bash
python analyze_results.py
```

This will generate:
1. Performance comparison plots
2. Speedup analysis
3. Efficiency analysis
4. Summary statistics

## Implementation Details

### parallel_for Library

The `parallel_for` implementation provides a high-level interface for parallel execution of loops, similar to OpenMP's parallel for construct. It handles:
- Thread creation and management
- Work distribution among threads
- Synchronization of thread completion

### Matrix Multiplication

The matrix multiplication implementation:
- Uses the parallel_for construct for row-wise parallelization
- Supports matrices of size 128x128 to 2048x2048
- Measures execution time for performance analysis

## Results

The analysis generates several plots:
1. `performance.png`: Shows execution time vs matrix size for different thread counts
2. `speedup.png`: Shows speedup vs number of threads for different matrix sizes
3. `efficiency.png`: Shows parallel efficiency vs number of threads

## License

This project is part of the Parallel Programming course lab assignments. 