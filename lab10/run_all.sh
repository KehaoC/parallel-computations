#!/bin/bash

# 创建结果目录
mkdir -p results

# 定义测试参数
MATRIX_SIZES=(128 512 1024 2048)
BLOCK_SIZES=(8 16 32)
IMPLEMENTATIONS=("naive" "shared")  # 可以根据实际实现添加更多

# 结果文件
RESULT_FILE="results/performance.csv"

# 创建结果文件头
echo "Implementation,MatrixSize,BlockSize,ExecutionTime(ms),Verified" > $RESULT_FILE

# 编译程序
echo "编译CUDA程序..."
nvcc -o src/gemm src/gemm.cu -O3

# 运行测试
echo "开始性能测试..."
for size in "${MATRIX_SIZES[@]}"; do
    echo "测试矩阵大小: ${size}x${size}"
    
    # 运行程序并捕获输出
    output=$(src/gemm $size $size $size)
    
    # 解析输出并提取性能数据
    echo "$output" > "results/output_${size}.txt"
    
    # 提取朴素实现的结果
    naive_results=$(echo "$output" | grep -A 4 "朴素CUDA矩阵乘法：" | grep -E "^[0-9]+x[0-9]+")
    
    # 提取共享内存实现的结果
    shared_results=$(echo "$output" | grep -A 4 "共享内存优化的CUDA矩阵乘法：" | grep -E "^[0-9]+x[0-9]+")
    
    # 处理朴素实现结果
    while IFS= read -r line; do
        if [[ ! -z "$line" ]]; then
            block_size=$(echo "$line" | awk '{print $1}' | cut -d 'x' -f 1)
            time=$(echo "$line" | awk '{print $2}')
            verified=$(echo "$line" | awk '{print $3}')
            echo "naive,$size,$block_size,$time,$verified" >> $RESULT_FILE
        fi
    done <<< "$naive_results"
    
    # 处理共享内存实现结果
    while IFS= read -r line; do
        if [[ ! -z "$line" ]]; then
            block_size=$(echo "$line" | awk '{print $1}' | cut -d 'x' -f 1)
            time=$(echo "$line" | awk '{print $2}')
            verified=$(echo "$line" | awk '{print $3}')
            echo "shared,$size,$block_size,$time,$verified" >> $RESULT_FILE
        fi
    done <<< "$shared_results"
done

echo "测试完成！结果已保存到 $RESULT_FILE"

# 生成简单的性能分析报告
echo "生成性能分析报告..."

# 创建性能分析报告文件
REPORT_FILE="results/performance_report.txt"

echo "CUDA矩阵乘法性能分析报告" > $REPORT_FILE
echo "==========================" >> $REPORT_FILE
echo "" >> $REPORT_FILE

# 分析不同矩阵大小的影响
echo "1. 矩阵规模对性能的影响" >> $REPORT_FILE
echo "--------------------------" >> $REPORT_FILE
for impl in "${IMPLEMENTATIONS[@]}"; do
    echo "实现方式: $impl" >> $REPORT_FILE
    echo "矩阵大小 | 平均执行时间(ms)" >> $REPORT_FILE
    for size in "${MATRIX_SIZES[@]}"; do
        avg_time=$(grep "$impl,$size," $RESULT_FILE | awk -F',' '{sum+=$4; count++} END {if(count>0) print sum/count; else print "N/A"}')
        echo "$size x $size | $avg_time" >> $REPORT_FILE
    done
    echo "" >> $REPORT_FILE
done

# 分析不同线程块大小的影响
echo "2. 线程块大小对性能的影响" >> $REPORT_FILE
echo "--------------------------" >> $REPORT_FILE
for impl in "${IMPLEMENTATIONS[@]}"; do
    echo "实现方式: $impl" >> $REPORT_FILE
    echo "线程块大小 | 平均执行时间(ms)" >> $REPORT_FILE
    for block in "${BLOCK_SIZES[@]}"; do
        avg_time=$(grep "$impl,[0-9]*,$block," $RESULT_FILE | awk -F',' '{sum+=$4; count++} END {if(count>0) print sum/count; else print "N/A"}')
        echo "${block}x${block} | $avg_time" >> $REPORT_FILE
    done
    echo "" >> $REPORT_FILE
done

# 比较不同实现方式的性能
echo "3. 不同实现方式的性能比较" >> $REPORT_FILE
echo "--------------------------" >> $REPORT_FILE
echo "矩阵大小 | 朴素实现(ms) | 共享内存实现(ms) | 性能提升(%)" >> $REPORT_FILE
for size in "${MATRIX_SIZES[@]}"; do
    naive_time=$(grep "naive,$size," $RESULT_FILE | awk -F',' '{sum+=$4; count++} END {if(count>0) print sum/count; else print "N/A"}')
    shared_time=$(grep "shared,$size," $RESULT_FILE | awk -F',' '{sum+=$4; count++} END {if(count>0) print sum/count; else print "N/A"}')
    
    # 计算性能提升百分比
    if [[ "$naive_time" != "N/A" && "$shared_time" != "N/A" ]]; then
        improvement=$(echo "scale=2; ($naive_time-$shared_time)/$naive_time*100" | bc)
        echo "$size x $size | $naive_time | $shared_time | $improvement%" >> $REPORT_FILE
    else
        echo "$size x $size | $naive_time | $shared_time | N/A" >> $REPORT_FILE
    fi
done

echo "" >> $REPORT_FILE
echo "测试环境:" >> $REPORT_FILE
echo "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> $REPORT_FILE

echo "性能分析报告已生成: $REPORT_FILE"

