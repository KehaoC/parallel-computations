#!/bin/bash

# 编译程序
make clean
make

# 矩阵大小数组
SIZES=(128 256 512 1024 2048)
# 进程数数组
PROCS=(1 2 4 8 16)

# 创建结果目录
mkdir -p results

# 运行实验
for size in "${SIZES[@]}"; do
    for proc in "${PROCS[@]}"; do
        echo "Running with matrix size $size and $proc processes..."
        # 使用--oversubscribe允许超过CPU核心数量的进程
        # 使用--use-hwthread-cpus使用硬件线程
        mpirun --oversubscribe --use-hwthread-cpus -np $proc ./build/matrix_mult $size 2>/dev/null | grep "Computation time" | awk '{print $3}' > results/results_${size}_${proc}.txt
    done
done

# 整理结果
echo "Matrix Size,Processes,Time" > results/summary.csv
for size in "${SIZES[@]}"; do
    for proc in "${PROCS[@]}"; do
        if [ -f "results/results_${size}_${proc}.txt" ]; then
            time=$(cat results/results_${size}_${proc}.txt)
            if [ ! -z "$time" ]; then
                echo "$size,$proc,$time" >> results/summary.csv
            else
                echo "$size,$proc,NA" >> results/summary.csv
            fi
        else
            echo "$size,$proc,NA" >> results/summary.csv
        fi
    done
done

echo "Experiments completed. Results saved in results/summary.csv" 