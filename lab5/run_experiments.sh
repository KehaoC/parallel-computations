#!/bin/bash

# Compile programs
make clean
make

# Matrix sizes array
SIZES=(128 256 512 1024 2048)
# Thread counts array
THREADS=(1 2 4 8 16)

# Create results directory
mkdir -p results

# Run Pthread experiments
echo "Running Pthread experiments..."
for size in "${SIZES[@]}"; do
    for thread in "${THREADS[@]}"; do
        echo "Running with matrix size $size and $thread threads..."
        ./build/matrix_mult_pthread $size $thread 2>/dev/null | grep "Computation time" | awk '{print $3}' > results/results_${size}_${thread}.txt
    done
done

# Create summary CSV file
echo "Matrix Size,Threads,Time" > results/summary.csv

# Process results
for size in "${SIZES[@]}"; do
    for thread in "${THREADS[@]}"; do
        if [ -f "results/results_${size}_${thread}.txt" ]; then
            time=$(cat results/results_${size}_${thread}.txt)
            if [ ! -z "$time" ]; then
                echo "$size,$thread,$time" >> results/summary.csv
            else
                echo "$size,$thread,NA" >> results/summary.csv
            fi
        else
            echo "$size,$thread,NA" >> results/summary.csv
        fi
    done
done

echo "Experiments completed. Results saved in results/summary.csv" 