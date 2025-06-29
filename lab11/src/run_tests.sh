#!/bin/bash

# 设置环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 编译程序
make clean
make

# 创建结果目录
mkdir -p ../results

# 测试不同大小的输入
echo "测试小尺寸输入 (32x32)"
./conv_test 32 3 | tee ../results/results_32.txt

echo "测试中等尺寸输入 (128x128)"
./conv_test 128 3 | tee ../results/results_128.txt

echo "测试大尺寸输入 (512x512)"
./conv_test 512 3 | tee ../results/results_512.txt

# 如果有足够的GPU内存，可以测试更大的输入
if [ "$1" == "large" ]; then
    echo "测试超大尺寸输入 (1024x1024)"
    ./conv_test 1024 3 | tee ../results/results_1024.txt
    
    echo "测试超大尺寸输入 (2048x2048)"
    ./conv_test 2048 3 | tee ../results/results_2048.txt
fi

echo "所有测试完成！" 