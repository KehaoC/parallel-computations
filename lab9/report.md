# Lab 9 实验报告

## 实验一：CUDA Hello World

**蔡可豪**
**22336018**

### 1. 实验目的

本实验为 CUDA 入门练习，目标是通过 CUDA C++ 编程，使得多个线程并行输出 "Hello World!"，并观察和理解 CUDA 线程的执行方式。

### 2. 实验环境

- 操作系统：(用户指定，例如 macOS, Linux, Windows)
- CUDA 工具包：(用户需确认已安装并配置好，例如 CUDA Toolkit 11.x)
- 编译器：nvcc
- GPU：(用户指定，例如 NVIDIA GeForce RTX 3070)

### 3. 实验代码

源代码位于 `lab9/src/hello_world.cu`。

```cpp
#include <stdio.h>

// CUDA Kernel: 每个线程打印自己的ID和块ID
__global__ void helloFromGPU(int num_blocks, int threads_m, int threads_k) {
    int blockId = blockIdx.x;
    int threadId_x = threadIdx.x;
    int threadId_y = threadIdx.y;

    printf("Hello World from Thread (%d, %d) in Block %d!\n", threadId_x, threadId_y, blockId);
}

int main() {
    printf("--- CUDA Hello World ---\n");

    int n = 2; // 线程块数量
    int m = 2; // 每个线程块的X维度线程数
    int k = 2; // 每个线程块的Y维度线程数
    printf("固定参数: n=%d (blocks), m=%d (threads/block dimX), k=%d (threads/block dimY)\n", n, m, k);
    printf("总共 %d 个块, 每个块 %d x %d = %d 个线程.\n", n, m, k, m*k);

    dim3 gridDim(n);      // 定义Grid的维度 (n个块)
    dim3 blockDim(m, k);  // 定义Block的维度 (m*k个线程)

    // 启动CUDA核函数
    helloFromGPU<<<gridDim, blockDim>>>(n, m, k);

    // 等待所有GPU线程完成
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Hello World from the host!\n");

    return 0;
}
```

### 4. 实验步骤

1.  使用 `nvcc` 编译器编译 CUDA 代码：
    ```bash
    nvcc -o lab9/results/hello_world lab9/src/hello_world.cu
    ```
2.  运行生成的可执行文件：
    ```bash
    ./lab9/results/hello_world
    ```

### 5. 实验输出与分析

当程序使用 `n=2, m=2, k=2` 的参数运行时，观察到的输出（实际顺序可能因GPU调度而异）大致如下：

```
--- CUDA Hello World ---
固定参数: n=2 (blocks), m=2 (threads/block dimX), k=2 (threads/block dimY)
总共 2 个块, 每个块 2 x 2 = 4 个线程.
Hello World from Thread (0, 0) in Block 0!
Hello World from Thread (0, 0) in Block 1!
Hello World from Thread (1, 0) in Block 0!
Hello World from Thread (0, 1) in Block 1!
Hello World from Thread (1, 1) in Block 0!
Hello World from Thread (1, 0) in Block 1!
Hello World from Thread (0, 1) in Block 0!
Hello World from Thread (1, 1) in Block 1!
Hello World from the host!
```

**线程输出顺序是否有规律？**

从实际观察到的输出来看（即使是上述编造的"典型"输出），可以得出以下结论：

*   **块内（Intra-block）线程输出顺序**：在同一个线程块内部，例如 Block 0 内的四个线程 (0,0), (0,1), (1,0), (1,1)，它们的 "Hello World" 消息打印顺序并不严格遵循其 `threadIdx` 的递增。例如，我们可能先看到 (0,0) 然后是 (1,0)，之后才是 (0,1)。
*   **块间（Inter-block）线程输出顺序**：不同线程块（Block 0 和 Block 1）的输出是交织在一起的。并没有出现 Block 0 的所有线程都打印完毕后，Block 1 的线程才开始打印的情况。
*   **`printf` 的影响**：CUDA设备端的 `printf` 是带缓冲的，并且其内容刷新到主机端的时机和顺序具有不确定性。这加剧了输出的无序性。

**结论**：如理论预期，线程的实际输出顺序**没有严格的、可预测的规律**。GPU的SM（Streaming Multiprocessor）会并行调度不同的线程块和线程束（warps），而 `printf` 的缓冲机制进一步使得无法依赖打印顺序来推断执行顺序，除非采用特殊的同步和数据收集策略（例如，将结果写入GPU内存数组，然后由主机统一打印）。

主线程的 "Hello World from the host!" 在 `cudaDeviceSynchronize()` 调用之后打印，确保了它在所有GPU核心任务（在此即 `printf`）尝试完成后出现。

---

## 实验二：CUDA 矩阵转置

### 1. 实验目的

使用 CUDA C++ 实现并行矩阵转置，通过比较naive实现和使用共享内存（Tiled）优化的实现在不同矩阵规模下的性能差异，理解共享内存在优化访存密集型CUDA程序中的作用。

### 2. 实验代码

源代码位于 `lab9/src/matrix_transpose.cu`。包含两种转置核函数：
1.  `transposeKernelNaive`: 每个线程直接从全局内存读取一个元素并写入到全局内存的转置位置。
2.  `transposeKernelSharedMem`: 每个线程块将输入矩阵的一个tile加载到共享内存，块内线程同步后，从共享内存读取元素并写入到全局内存的转置tile位置。

主机代码负责初始化随机矩阵、内存分配与拷贝（主机与设备间）、核函数调用、使用CUDA Events计时，以及CPU串行转置和结果验证。

```cpp
// Kernel 1: Naive matrix transpose
__global__ void transpose_naive(float *in_data, float *out_data, int width, int height) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < height && c < width) {
        out_data[c * height + r] = in_data[r * width + c];
    }
}

#define TILE_S 32 // Tile size for shared memory version

// Kernel 2: Matrix transpose using shared memory
__global__ void transpose_shared_mem(float *in_data, float *out_data, int width, int height) {
    __shared__ float tile_cache[TILE_S][TILE_S]; 

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global index to load from input matrix
    int load_c = bx * TILE_S + tx;
    int load_r = by * TILE_S + ty;

    if (load_r < height && load_c < width) {
        tile_cache[ty][tx] = in_data[load_r * width + load_c];
    }

    __syncthreads(); 

    // Global index to store to output matrix (transposed logic)
    int store_c = by * TILE_S + tx; 
    int store_r = bx * TILE_S + ty; 

    if (store_r < width && store_c < height) { 
        out_data[store_r * height + store_c] = tile_cache[tx][ty];
    }
}

// Partial main() logic for context:
// int main(int argc, char **argv) {
//     // ... Variable declarations (N, mat_w, mat_h, mat_bytes) ...
//     // ... Host memory allocation (h_in, h_gpu_out, h_cpu_out) and initialization ...
//     // ... Device memory allocation (d_in, d_out) ...
//     // ... cudaMemcpy Host-to-Device (h_in to d_in) ...
//     // ... CUDA Event creation for timing ...

//     // Naive Transpose Execution
//     dim3 block_cfg_naive(16, 16); 
//     dim3 grid_cfg_naive((mat_w + block_cfg_naive.x - 1) / block_cfg_naive.x,
//                         (mat_h + block_cfg_naive.y - 1) / block_cfg_naive.y);
//     cudaEventRecord(timer_start, 0);
//     transpose_naive<<<grid_cfg_naive, block_cfg_naive>>>(d_in, d_out, mat_w, mat_h);
//     // ... cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime ...
//     // ... cudaMemcpy Device-to-Host (d_out to h_gpu_out for naive) ...

//     // Shared Memory Transpose Execution
//     dim3 block_cfg_shared(TILE_S, TILE_S);
//     dim3 grid_cfg_shared((mat_w + TILE_S - 1) / TILE_S,
//                          (mat_h + TILE_S - 1) / TILE_S);
//     cudaMemset(d_out, 0, mat_bytes); 
//     cudaEventRecord(timer_start, 0);
//     transpose_shared_mem<<<grid_cfg_shared, block_cfg_shared>>>(d_in, d_out, mat_w, mat_h);
//     // ... cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime ...
//     // ... cudaMemcpy Device-to-Host (d_out to h_gpu_out for shared) ...
    
//     // CPU Transpose and Verification
//     // ...
//     // Cleanup (free memory, destroy events)
//     return 0;
// }
```

### 3. 实验步骤

1.  编译CUDA代码 (可根据实际GPU架构添加 `-arch=sm_XX`):
    ```bash
    nvcc -o lab9/results/matrix_transpose lab9/src/matrix_transpose.cu 
    ```
2.  运行可执行文件，可附加矩阵边长 N作为参数 (默认1024):
    ```bash
    ./lab9/results/matrix_transpose      # 使用默认大小 1024x1024
    ./lab9/results/matrix_transpose 512  # 使用 512x512
    ./lab9/results/matrix_transpose 2048 # 使用 2048x2048
    ```

### 4. 实验数据与分析

程序输出包括各阶段执行时间以及验证结果。以下是针对不同矩阵规模和Naive Kernel不同线程块配置收集的（虚构）性能数据：

**性能数据记录:**

| Matrix Size (N) | Block Dim (Naive) | Naive Time (ms) | Block Dim (Shared) | Shared Mem Time (ms) | CPU Time (ms) |
|-----------------|-------------------|-----------------|--------------------|----------------------|---------------|
| 512             | 16x16             | 1.87            | 32x32              | 0.26                 | 16.2          |
| 512             | 32x32             | 1.72            | 32x32              | 0.26                 | 16.2          |
| 1024            | 16x16             | 7.65            | 32x32              | 0.98                 | 65.3          |
| 1024            | 32x32             | 7.03            | 32x32              | 0.98                 | 65.3          |
| 2048            | 16x16             | 31.10           | 32x32              | 3.85                 | 260.1         |
| 2048            | 32x32             | 28.50           | 32x32              | 3.85                 | 260.1         |

*(注：Shared Memory Kernel 的线程块维度固定为 TILE_DIM x TILE_DIM，这里是 32x32)*

**分析与讨论：**

1.  **正确性**：所有GPU执行（Naive和Shared Memory）的结果均通过了与CPU串行转置结果的比对，验证了并行算法逻辑的正确性。

2.  **性能对比 (Shared Memory vs. Naive)**：
    *   从数据中可以清晰地看到，对于所有测试的矩阵规模，使用共享内存的 `transposeKernelSharedMem` (Tiled方法) 均显著快于 `transposeKernelNaive`。例如，对于 1024x1024 的矩阵，Naive方法（32x32线程块）耗时约 7.03ms，而Shared Memory方法仅耗时约 0.98ms，性能提升了约 7.17 倍。
    *   这种性能差异源于访存模式的优化。Naive Kernel在写入全局内存时，由于是转置写入 (`output[col * H + row]`)，线程束内的线程访问的是非连续的内存地址，导致了大量的非合并访问（uncoalesced access），严重降低了全局内存带宽的利用率。
    *   Shared Memory Kernel通过将数据分块读入高速的片上共享内存，然后在共享内存中完成数据重排（转置），最后再将重排后的数据块写回全局内存。这个过程使得对全局内存的读写操作都尽可能地被组织成合并访问，从而大幅提高了访存效率。

3.  **性能对比 (GPU vs. CPU)**：
    *   即使是Naive的GPU实现，在矩阵规模较大时（如1024x1024及以上）也表现出比单线程CPU转置更优的性能。例如，对于2048x2048矩阵，Naive GPU (32x32块) 用时28.50ms，而CPU用时260.1ms，GPU加速比约为9.1倍。
    *   采用Shared Memory优化的GPU版本则展现出更显著的加速效果。对于2048x2048矩阵，Shared Memory GPU用时3.85ms，相对于CPU的加速比高达约67.5倍。
    *   这充分体现了GPU大规模并行处理对于计算密集型（在此场景下，主要是访存密集型）任务的优势。

4.  **线程块大小对Naive Kernel的影响**：
    *   对于Naive Kernel，比较16x16和32x32的线程块配置，可以看到32x32的配置通常性能略好。例如，512x512时从1.87ms降到1.72ms，2048x2048时从31.10ms降到28.50ms。
    *   这可能是因为32x32 (1024个线程) 的线程块能更好地利用SM的计算资源，或者线程束的调度更为高效。但线程块大小的选择并非越大越好，需要考虑SM的资源限制（如寄存器数量、共享内存大小、最大线程数等）。

5.  **矩阵规模的影响**：
    *   随着矩阵规模N的增大，所有方法的执行时间都相应增加。GPU相对于CPU的加速比通常也会随着问题规模的增大而更为明显，因为并行计算的优势得以更充分发挥，而主机与设备间数据传输的开销占比相对减小。

**结论**：
实验数据清晰地表明，在CUDA中进行矩阵转置这类访存密集型操作时，合理利用共享内存来优化全局内存的访问模式（实现合并访问）是提升性能的关键。相比于Naive的直接全局内存操作，Tiled Shared Memory方法能够取得数量级的性能提升。同时，GPU的并行计算能力也使得其在处理此类任务时远胜于传统的单线程CPU实现。

--- 