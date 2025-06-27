#!/bin/bash

# 创建结果目录
mkdir -p results

# 矩阵乘法实验
echo "Running matrix multiplication experiments..."
matrix_sizes=(128 256 512 1024 2048)
thread_counts=(1 2 4 8 16)

for size in "${matrix_sizes[@]}"; do
    for threads in "${thread_counts[@]}"; do
        echo "Matrix size: $size x $size, Threads: $threads"
        ./build/matrix_multiply $size $threads >> "results/matrix_${size}_${threads}.txt"
    done
done

# 数组求和实验
echo "Running array sum experiments..."
# 1M = 1048576, 2M = 2097152, 4M = 4194304, 8M = 8388608, 16M = 16777216, 32M = 33554432, 64M = 67108864, 128M = 134217728
array_sizes=(1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728)

for size in "${array_sizes[@]}"; do
    for threads in "${thread_counts[@]}"; do
        echo "Array size: $size, Threads: $threads"
        ./build/array_sum $size $threads >> "results/array_${size}_${threads}.txt"
    done
done

echo "All experiments completed!" 