import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the results
df = pd.read_csv('results/summary.csv')

# Create output directory for plots
import os
os.makedirs('results/plots', exist_ok=True)

# Plot performance comparison
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Matrix Size', y='Time', hue='Threads', marker='o')
plt.title('Matrix Multiplication Performance')
plt.xlabel('Matrix Size')
plt.ylabel('Computation Time (seconds)')
plt.grid(True)
plt.savefig('results/plots/performance.png')
plt.close()

# Calculate speedup for each matrix size
speedups = []
for size in df['Matrix Size'].unique():
    size_df = df[df['Matrix Size'] == size]
    base_time = size_df[size_df['Threads'] == 1]['Time'].values[0]
    for _, row in size_df.iterrows():
        speedups.append({
            'Matrix Size': size,
            'Threads': row['Threads'],
            'Speedup': base_time / row['Time']
        })
speedup_df = pd.DataFrame(speedups)

# Plot speedup
plt.figure(figsize=(12, 8))
for size in speedup_df['Matrix Size'].unique():
    size_speedup = speedup_df[speedup_df['Matrix Size'] == size]
    plt.plot(size_speedup['Threads'], size_speedup['Speedup'], 
             marker='o', label=f'Size {size}')

plt.title('Speedup vs Number of Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.grid(True)
plt.legend()
plt.savefig('results/plots/speedup.png')
plt.close()

# Calculate efficiency
efficiency_df = speedup_df.copy()
efficiency_df['Efficiency'] = efficiency_df['Speedup'] / efficiency_df['Threads']

# Plot efficiency
plt.figure(figsize=(12, 8))
for size in efficiency_df['Matrix Size'].unique():
    size_efficiency = efficiency_df[efficiency_df['Matrix Size'] == size]
    plt.plot(size_efficiency['Threads'], size_efficiency['Efficiency'], 
             marker='o', label=f'Size {size}')

plt.title('Efficiency vs Number of Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Efficiency')
plt.grid(True)
plt.legend()
plt.savefig('results/plots/efficiency.png')
plt.close()

# Generate summary statistics
summary_stats = df.groupby(['Matrix Size', 'Threads']).agg({
    'Time': ['mean', 'std', 'min', 'max']
}).reset_index()

summary_stats.columns = ['Matrix Size', 'Threads', 'Mean Time', 'Std Time', 'Min Time', 'Max Time']
summary_stats.to_csv('results/summary_statistics.csv', index=False)

print("Analysis complete. Plots and statistics have been saved to the results directory.") 