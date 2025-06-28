
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取性能数据
df = pd.read_csv('performance.csv')

# 设置图表风格
plt.style.use('ggplot')
plt.figure(figsize=(15, 10))

# 1. 矩阵大小对性能的影响
plt.subplot(2, 2, 1)
for impl in df['Implementation'].unique():
    for block in df['BlockSize'].unique():
        data = df[(df['Implementation'] == impl) & (df['BlockSize'] == block)]
        plt.plot(data['MatrixSize'], data['ExecutionTime(ms)'], 
                 marker='o', label=f"{impl}, {block}x{block}")

plt.title('矩阵大小对执行时间的影响')
plt.xlabel('矩阵大小')
plt.ylabel('执行时间 (ms)')
plt.xscale('log2')
plt.yscale('log')
plt.legend()
plt.grid(True)

# 2. 线程块大小对性能的影响
plt.subplot(2, 2, 2)
for impl in df['Implementation'].unique():
    for size in df['MatrixSize'].unique():
        data = df[(df['Implementation'] == impl) & (df['MatrixSize'] == size)]
        plt.plot(data['BlockSize'], data['ExecutionTime(ms)'], 
                 marker='o', label=f"{impl}, {size}x{size}")

plt.title('线程块大小对执行时间的影响')
plt.xlabel('线程块大小')
plt.ylabel('执行时间 (ms)')
plt.legend()
plt.grid(True)

# 3. 不同实现方式的性能比较
plt.subplot(2, 2, 3)
implementations = df['Implementation'].unique()
matrix_sizes = df['MatrixSize'].unique()

x = np.arange(len(matrix_sizes))
width = 0.35 / len(implementations)

for i, impl in enumerate(implementations):
    times = []
    for size in matrix_sizes:
        avg_time = df[(df['Implementation'] == impl) & (df['MatrixSize'] == size)]['ExecutionTime(ms)'].mean()
        times.append(avg_time)
    
    plt.bar(x + i*width, times, width, label=impl)

plt.title('不同实现方式的性能比较')
plt.xlabel('矩阵大小')
plt.ylabel('平均执行时间 (ms)')
plt.xticks(x, [f"{size}x{size}" for size in matrix_sizes])
plt.legend()

# 4. 性能加速比
plt.subplot(2, 2, 4)
if len(implementations) > 1:
    speedups = []
    for size in matrix_sizes:
        base_time = df[(df['Implementation'] == 'naive') & (df['MatrixSize'] == size)]['ExecutionTime(ms)'].mean()
        for impl in implementations:
            if impl != 'naive':
                opt_time = df[(df['Implementation'] == impl) & (df['MatrixSize'] == size)]['ExecutionTime(ms)'].mean()
                speedup = base_time / opt_time if opt_time > 0 else 0
                speedups.append((size, impl, speedup))
    
    # 绘制加速比
    for impl in implementations:
        if impl != 'naive':
            impl_data = [(s, sp) for s, i, sp in speedups if i == impl]
            plt.plot([s for s, _ in impl_data], [sp for _, sp in impl_data], marker='o', label=impl)
    
    plt.title('不同实现方式相对于朴素实现的加速比')
    plt.xlabel('矩阵大小')
    plt.ylabel('加速比')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('performance_plots.png', dpi=300)
plt.close()

print("性能图表已生成: performance_plots.png")
