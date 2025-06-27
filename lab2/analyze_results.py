import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取数据
try:
    df = pd.read_csv('results/summary.csv')
except FileNotFoundError:
    print("Error: results/summary.csv not found. Please run the experiments first.")
    exit(1)

# 创建图表
plt.figure(figsize=(12, 8))

# 为不同矩阵大小创建子图
matrix_sizes = sorted(df['Matrix Size'].unique())
num_sizes = len(matrix_sizes)

# 设置颜色方案
colors = sns.color_palette("husl", n_colors=len(matrix_sizes))

# 创建加速比图
plt.figure(figsize=(10, 6))
for i, size in enumerate(matrix_sizes):
    size_data = df[df['Matrix Size'] == size]
    if len(size_data) > 0 and 'NA' not in size_data['Time'].values:
        sequential_time = float(size_data[size_data['Processes'] == 1]['Time'].values[0])
        speedup = sequential_time / size_data['Time'].astype(float)
        plt.plot(size_data['Processes'], speedup, 'o-', label=f'{size}x{size}', color=colors[i])

plt.plot([1, max(df['Processes'])], [1, max(df['Processes'])], '--', label='Ideal', color='gray')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.title('Speedup vs Number of Processes')
plt.grid(True)
plt.legend()
plt.savefig('results/speedup_analysis.png')

# 创建效率图
plt.figure(figsize=(10, 6))
for i, size in enumerate(matrix_sizes):
    size_data = df[df['Matrix Size'] == size]
    if len(size_data) > 0 and 'NA' not in size_data['Time'].values:
        sequential_time = float(size_data[size_data['Processes'] == 1]['Time'].values[0])
        speedup = sequential_time / size_data['Time'].astype(float)
        efficiency = speedup / size_data['Processes'].astype(float)
        plt.plot(size_data['Processes'], efficiency, 'o-', label=f'{size}x{size}', color=colors[i])

plt.xlabel('Number of Processes')
plt.ylabel('Parallel Efficiency')
plt.title('Parallel Efficiency vs Number of Processes')
plt.grid(True)
plt.legend()
plt.savefig('results/efficiency_analysis.png')

# 打印详细结果
print("\nDetailed Performance Analysis:")
print("=" * 80)
for size in matrix_sizes:
    size_data = df[df['Matrix Size'] == size]
    if len(size_data) > 0:
        print(f"\nMatrix Size: {size}x{size}")
        print("-" * 40)
        print("Processes  Time(s)  Speedup  Efficiency")
        print("-" * 40)
        
        # 获取顺序执行时间
        sequential_data = size_data[size_data['Processes'] == 1]
        if len(sequential_data) > 0 and sequential_data['Time'].values[0] != 'NA':
            sequential_time = float(sequential_data['Time'].values[0])
            
            for _, row in size_data.iterrows():
                if row['Time'] != 'NA':
                    time = float(row['Time'])
                    processes = int(row['Processes'])
                    speedup = sequential_time / time
                    efficiency = speedup / processes
                    print(f"{processes:^9d} {time:^8.3f} {speedup:^8.2f} {efficiency:^10.2f}")
                else:
                    print(f"{int(row['Processes']):^9d} {'NA':^8s} {'NA':^8s} {'NA':^10s}") 