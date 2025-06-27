#!/bin/bash

# Matrix sizes to test
SIZES=(128 256 512 1024 2048)
# Process counts to test
PROCS=(1 2 4 8)

# Create results directory if it doesn't exist
mkdir -p results

# Create CSV file for results
echo "Processes,Size,Time" > results/timing_results.csv

# Run experiments
for size in "${SIZES[@]}"; do
    for proc in "${PROCS[@]}"; do
        echo "Running with $proc processes and size $size x $size"
        # Run the program and extract the time
        output=$(mpirun -np $proc ./matrix_multiply $size $size $size)
        time=$(echo "$output" | grep "Time taken:" | awk '{print $3}')
        # Append to CSV
        echo "$proc,$size,$time" >> results/timing_results.csv
    done
done

# Print results in a table format
echo -e "\nResults (time in seconds):"
echo "Process Count | 128 | 256 | 512 | 1024 | 2048"
echo "-------------|-----|-----|-----|------|------"
for proc in "${PROCS[@]}"; do
    printf "%12d |" $proc
    for size in "${SIZES[@]}"; do
        time=$(grep "^$proc,$size," results/timing_results.csv | cut -d',' -f3)
        printf " %0.3f |" $time
    done
    echo
done 