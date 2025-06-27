import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_quadratic_results():
    data = []
    for threads in [1, 2, 4, 8, 16]:
        filename = f'results/quadratic_{threads}.txt'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
                # Count how many equations were solved
                equations = len(re.findall(r'Thread \d+:', content))
                data.append({'threads': threads, 'equations': equations})
    
    return pd.DataFrame(data)

def parse_monte_carlo_results():
    data = []
    # Parse results for fixed points, varying threads
    points = 65536
    for threads in [1, 2, 4, 8, 16]:
        filename = f'results/monte_carlo_{threads}_{points}.txt'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
                pi_estimate = float(re.search(r'Pi estimate: ([\d.]+)', content).group(1))
                error = float(re.search(r'Absolute error: ([\d.]+)', content).group(1))
                time = float(re.search(r'Execution time: ([\d.]+)', content).group(1))
                data.append({
                    'threads': threads,
                    'points': points,
                    'pi_estimate': pi_estimate,
                    'error': error,
                    'time': time
                })
    
    # Parse results for fixed threads, varying points
    threads = 8
    for points in [1024, 4096, 16384, 65536]:
        filename = f'results/monte_carlo_{threads}_{points}.txt'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
                pi_estimate = float(re.search(r'Pi estimate: ([\d.]+)', content).group(1))
                error = float(re.search(r'Absolute error: ([\d.]+)', content).group(1))
                time = float(re.search(r'Execution time: ([\d.]+)', content).group(1))
                data.append({
                    'threads': threads,
                    'points': points,
                    'pi_estimate': pi_estimate,
                    'error': error,
                    'time': time
                })
    
    return pd.DataFrame(data)

def plot_results():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot quadratic solver results
    quad_df = parse_quadratic_results()
    plt.figure(figsize=(10, 6))
    plt.plot(quad_df['threads'], quad_df['equations'], 'o-')
    plt.title('Quadratic Equations Solved vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Number of Equations Solved')
    plt.grid(True)
    plt.savefig('results/quadratic_analysis.png')
    plt.close()
    
    # Plot Monte Carlo results
    mc_df = parse_monte_carlo_results()
    
    # Plot execution time vs threads (fixed points)
    fixed_points_df = mc_df[mc_df['points'] == 65536]
    plt.figure(figsize=(10, 6))
    plt.plot(fixed_points_df['threads'], fixed_points_df['time'], 'o-')
    plt.title('Execution Time vs Number of Threads (65536 points)')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig('results/monte_carlo_time_vs_threads.png')
    plt.close()
    
    # Plot error vs points (fixed threads)
    fixed_threads_df = mc_df[mc_df['threads'] == 8]
    plt.figure(figsize=(10, 6))
    plt.plot(fixed_threads_df['points'], fixed_threads_df['error'], 'o-')
    plt.title('Error vs Number of Points (8 threads)')
    plt.xlabel('Number of Points')
    plt.ylabel('Absolute Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('results/monte_carlo_error_vs_points.png')
    plt.close()
    
    # Save numerical results
    mc_df.to_csv('results/monte_carlo_summary.csv', index=False)
    quad_df.to_csv('results/quadratic_summary.csv', index=False)

if __name__ == '__main__':
    plot_results() 