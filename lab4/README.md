# 并行计算实验 4

本实验包含两个子实验：
1. 使用 Pthread 多线程实现一元二次方程求解
2. 使用蒙特卡洛方法并行计算圆周率 π 的近似值

## 项目结构

```
lab4/
├── src/                    # 源代码目录
│   ├── quadratic_solver.c  # 一元二次方程求解程序
│   └── monte_carlo_pi.c    # 蒙特卡洛求π程序
├── build/                  # 编译输出目录
├── results/               # 实验结果目录
├── Makefile              # 编译脚本
├── run_experiments.sh    # 实验运行脚本
└── analyze_results.py    # 数据分析脚本
```

## 编译和运行

1. 编译项目：
```bash
make
```

2. 运行实验：
```bash
./run_experiments.sh
```

3. 分析结果：
```bash
python3 analyze_results.py
```

## 实验内容

### 实验1：一元二次方程求解

使用 Pthread 多线程实现一元二次方程求解，主要特点：
- 使用多线程并行计算不同的方程
- 使用条件变量识别线程完成情况
- 分析计算任务依赖关系及程序并行性能

### 实验2：蒙特卡洛求π

使用蒙特卡洛方法并行计算圆周率π的近似值：
- 使用 Pthread 创建多线程，并行生成随机点
- 统计落在单位正方形内切圆内的点数
- 设置不同的线程数量（1-16）和随机点数量（1024-65536）
- 分析近似精度和程序并行性能

## 实验结果

实验结果将保存在 `results/` 目录下：
- `quadratic_analysis.png`：一元二次方程求解性能分析
- `monte_carlo_time_vs_threads.png`：蒙特卡洛方法的执行时间与线程数关系
- `monte_carlo_error_vs_points.png`：蒙特卡洛方法的误差与点数关系
- `monte_carlo_summary.csv`：蒙特卡洛实验数据汇总
- `quadratic_summary.csv`：一元二次方程实验数据汇总 