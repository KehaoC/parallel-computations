#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Experiment 1: Quadratic Equation Solver
echo "Running Quadratic Equation Solver experiments..."
for threads in 1 2 4 8 16
do
    echo "Testing with $threads threads..."
    ./build/quadratic_solver $threads > results/quadratic_${threads}.txt
done

# Experiment 2: Monte Carlo Pi Estimation
echo -e "\nRunning Monte Carlo Pi experiments..."
# Test different thread counts with fixed number of points
points=65536
for threads in 1 2 4 8 16
do
    echo "Testing with $threads threads and $points points..."
    ./build/monte_carlo_pi $threads $points > results/monte_carlo_${threads}_${points}.txt
done

# Test different point counts with fixed number of threads
threads=8
for points in 1024 4096 16384 65536
do
    echo "Testing with $threads threads and $points points..."
    ./build/monte_carlo_pi $threads $points > results/monte_carlo_${threads}_${points}.txt
done

echo "All experiments completed. Results are in the results directory." 