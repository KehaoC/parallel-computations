import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def parse_result_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        serial_time = float(re.search(r'Serial time: ([\d.]+)', content).group(1))
        parallel_time = float(re.search(r'Parallel time \(\d+ threads\): ([\d.]+)', content).group(1))
        return serial_time, parallel_time

def analyze_matrix_results():
    results_dir = Path('results')
    matrix_sizes = [128, 256, 512, 1024, 2048]
    thread_counts = [1, 2, 4, 8, 16]
    
    speedups = np.zeros((len(matrix_sizes), len(thread_counts)))
    
    for i, size in enumerate(matrix_sizes):
        for j, threads in enumerate(thread_counts):
            file_path = results_dir / f'matrix_{size}_{threads}.txt'
            serial_time, parallel_time = parse_result_file(file_path)
            speedups[i, j] = serial_time / parallel_time
    
    # 绘制加速比图
    plt.figure(figsize=(10, 6))
    for i, size in enumerate(matrix_sizes):
        plt.plot(thread_counts, speedups[i], marker='o', label=f'{size}x{size}')
    
    plt.xlabel('线程数')
    plt.ylabel('加速比')
    plt.title('矩阵乘法并行加速比')
    plt.grid(True)
    plt.legend()
    plt.savefig('report/matrix_speedup.png')
    plt.close()
    
    # 生成效率表格
    efficiency = speedups / np.array(thread_counts)
    df = pd.DataFrame(efficiency, 
                     index=[f'{s}x{s}' for s in matrix_sizes],
                     columns=[f'{t} threads' for t in thread_counts])
    df.to_csv('report/matrix_efficiency.csv')
    
    return speedups, efficiency

def analyze_array_results():
    results_dir = Path('results')
    array_sizes = [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728]
    thread_counts = [1, 2, 4, 8, 16]
    
    speedups = np.zeros((len(array_sizes), len(thread_counts)))
    
    for i, size in enumerate(array_sizes):
        for j, threads in enumerate(thread_counts):
            file_path = results_dir / f'array_{size}_{threads}.txt'
            serial_time, parallel_time = parse_result_file(file_path)
            speedups[i, j] = serial_time / parallel_time
    
    # 绘制加速比图
    plt.figure(figsize=(10, 6))
    for i, size in enumerate(array_sizes):
        plt.plot(thread_counts, speedups[i], marker='o', label=f'{size//1048576}M')
    
    plt.xlabel('线程数')
    plt.ylabel('加速比')
    plt.title('数组求和并行加速比')
    plt.grid(True)
    plt.legend()
    plt.savefig('report/array_speedup.png')
    plt.close()
    
    # 生成效率表格
    efficiency = speedups / np.array(thread_counts)
    df = pd.DataFrame(efficiency,
                     index=[f'{s//1048576}M' for s in array_sizes],
                     columns=[f'{t} threads' for t in thread_counts])
    df.to_csv('report/array_efficiency.csv')
    
    return speedups, efficiency

def main():
    os.makedirs('report', exist_ok=True)
    
    print("Analyzing matrix multiplication results...")
    matrix_speedups, matrix_efficiency = analyze_matrix_results()
    
    print("Analyzing array sum results...")
    array_speedups, array_efficiency = analyze_array_results()
    
    # 生成Markdown报告
    with open('report/README.md', 'w') as f:
        f.write('# 并行计算实验报告\n\n')
        
        f.write('## 实验1：矩阵乘法\n\n')
        f.write('### 加速比分析\n\n')
        f.write('![矩阵乘法加速比](matrix_speedup.png)\n\n')
        f.write('### 并行效率分析\n\n')
        f.write('详细数据见 matrix_efficiency.csv\n\n')
        
        f.write('## 实验2：数组求和\n\n')
        f.write('### 加速比分析\n\n')
        f.write('![数组求和加速比](array_speedup.png)\n\n')
        f.write('### 并行效率分析\n\n')
        f.write('详细数据见 array_efficiency.csv\n\n')

if __name__ == '__main__':
    main() 