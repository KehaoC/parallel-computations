#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <mpi.h>
#include <numeric> // For std::iota
#include <algorithm> // For std::copy

const double PI = acos(-1.0);

// 串行FFT函数 (与fft_serial.cpp中的fft函数类似，但可能需要微调以适应MPI上下文)
// 为了简化，这里直接复制 fft_serial.cpp 中的 fft 函数。
// 实际应用中，如果 fft_serial.cpp 的实现很复杂，可能需要重构或调用其函数。
void fft_recursive(std::vector<std::complex<double> >& a, bool invert) {
    int n = a.size();
    if (n <= 1) return;

    std::vector<std::complex<double> > a0(n / 2), a1(n / 2);
    for (int i = 0, j = 0; i < n; i += 2, ++j) {
        a0[j] = a[i];
        a1[j] = a[i + 1];
    }

    fft_recursive(a0, invert);
    fft_recursive(a1, invert);

    double ang = 2 * PI / n * (invert ? -1 : 1);
    std::complex<double> w(1), wn(cos(ang), sin(ang));

    for (int i = 0; i < n / 2; ++i) {
        std::complex<double> t = w * a1[i];
        a[i] = a0[i] + t;
        a[i + n / 2] = a0[i] - t;
        w *= wn;
    }

    if (invert) {
        for (int i = 0; i < n; ++i) {
            a[i] /= 2;
        }
    }
}


// 主函数
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_global = 32; // 全局输入大小，必须是2的幂，且能被进程数整除
    if (argc > 1) {
        n_global = std::atoi(argv[1]);
    }

    if (rank == 0) {
        if (n_global <= 0 || (n_global & (n_global - 1)) != 0) {
            std::cerr << "Global input size N must be a power of 2." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (n_global % size != 0) {
            std::cerr << "Global input size N must be divisible by the number of MPI processes." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 广播 n_global, 以防有进程因参数问题提前退出
    MPI_Bcast(&n_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n_global <= 0 || (n_global & (n_global - 1)) != 0 || n_global % size != 0) {
        if (rank != 0) { // rank 0 已经打印过错误
             // 其他进程等待rank 0 Abort
        }
        MPI_Finalize();
        return 1;
    }


    int n_local = n_global / size; // 每个进程处理的数据大小

    std::vector<std::complex<double> > local_data(n_local);
    std::vector<std::complex<double> > global_data;
    std::vector<std::complex<double> > global_result_fft;

    if (rank == 0) {
        global_data.resize(n_global);
        global_result_fft.resize(n_global); // 用于存放最终的FFT结果
        std::cout << "MPI FFT on " << size << " processes." << std::endl;
        std::cout << "Global input data (size " << n_global << "):" << std::endl;
        for (int i = 0; i < n_global; ++i) {
            global_data[i] = std::complex<double>(rand() % 100, 0); // 简单生成数据
            // global_data[i] = std::complex<double>(i, 0); // 或者使用确定性数据
             std::cout << global_data[i] << " ";
        }
        std::cout << std::endl << std::endl;
    }

    // 分发数据: 主进程使用 MPI_Scatter 将 global_data 分发到所有进程的 local_data
    // 注意: MPI_Scatter 对于 std::vector<std::complex<double>> 可能需要自定义 MPI_Datatype
    // 或者更简单的方式是，主进程发送，其他进程接收。
    // 为了简单起见，这里使用 MPI_Scatter。需要确保 std::complex<double> 可以直接传递。
    // MPI 定义了 MPI_DOUBLE_COMPLEX 类型。
    MPI_Scatter(global_data.data(), n_local, MPI_DOUBLE_COMPLEX,
                local_data.data(), n_local, MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

    // --- 并行 FFT 计算 ---
    // 这是一个简化的示例，真正的并行 FFT 会更复杂。
    // 策略1: 每个进程独立计算其 local_data 的 FFT (这不是完整的 FFT)
    // fft_recursive(local_data, false); // 这只是对分块数据的局部FFT

    // 策略2: 尝试实现一个基于蝶形运算的简单并行FFT (Cooley-Tukey的迭代版或递归版并行化)
    // 对于迭代算法（如Cooley-Tukey），并行化通常涉及：
    // 1. 数据初始分布（如位反转后分块）
    // 2. 迭代计算蝶形：
    //    - log(N) 个阶段
    //    - 每个阶段内，一些蝶形运算是局部的，一些需要跨进程通信

    // 为了"能跑起来"，我们先用一个非常简化的方法：
    // 每个进程对其本地数据进行FFT，然后收集。这不是一个正确的分布式FFT，
    // 但能让我们搭建起MPI的框架。之后再改进。
    // 正确的分布式FFT需要仔细处理数据交换。

    // 简化版：每个进程做局部FFT
    fft_recursive(local_data, false);


    // 收集结果: 主进程使用 MPI_Gather 将所有进程的 local_data (已FFT) 收集到 global_result_fft
    MPI_Gather(local_data.data(), n_local, MPI_DOUBLE_COMPLEX,
               global_result_fft.data(), n_local, MPI_DOUBLE_COMPLEX,
               0, MPI_COMM_WORLD);

    // 在实际的并行FFT中，MPI_Gather 之后，数据可能不是最终的FFT顺序，
    // 可能还需要进一步的全局重排或者在之前的阶段有更复杂的通信模式（如Alltoall）。
    // 对于Cooley-Tukey算法，如果数据按进程分块，那么在log(size)个阶段需要Alltoall通信。

    if (rank == 0) {
        std::cout << "Gathered (partially processed) FFT result:" << std::endl;
        for (int i = 0; i < n_global; ++i) {
            std::cout << global_result_fft[i] << " ";
        }
        std::cout << std::endl << std::endl;

        // ---- 如果要验证，需要一个串行FFT作为对照 ----
        // std::vector<std::complex<double>> verification_fft = global_data; // 复制原始数据
        // fft_recursive(verification_fft, false); // 用串行版本计算
        // std::cout << "Serial FFT for verification:" << std::endl;
        // for (int i = 0; i < n_global; ++i) {
        //     std::cout << verification_fft[i] << " ";
        // }
        // std::cout << std::endl;
        
        // ---- 接下来进行 IFFT ----
        // 为了演示完整的流程，我们将收集到的 (可能不完全正确的) FFT 结果进行 IFFT
        // 首先将 global_result_fft 再次分发
        std::copy(global_result_fft.begin(), global_result_fft.end(), global_data.begin()); // 复用 global_data 作为发送缓冲区
    }
    
    // 广播 IFFT 的输入数据
    // 所有进程都需要为接收 Bcast 数据准备空间
    if (rank != 0) { // rank 0 的 global_data 已经有数据，并且大小正确
        global_data.resize(n_global);
    }
    MPI_Bcast(global_data.data(), n_global, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    // 每个进程拿到完整的 (部分处理过的) FFT 结果的一部分
     std::copy(global_data.begin() + rank * n_local, 
              global_data.begin() + (rank + 1) * n_local, 
              local_data.begin());


    // 每个进程对其本地数据进行 IFFT
    fft_recursive(local_data, true);

    // 收集 IFFT 结果
    std::vector<std::complex<double> > global_result_ifft;
    if (rank == 0) {
        global_result_ifft.resize(n_global);
    }
    MPI_Gather(local_data.data(), n_local, MPI_DOUBLE_COMPLEX,
               global_result_ifft.data(), n_local, MPI_DOUBLE_COMPLEX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\nDistributed IFFT result (should be close to input if FFT was correct):" << std::endl;
        for (int i = 0; i < n_global; ++i) {
            // 因为IFFT后每个元素都除了N (在递归中是每层除2，总共logN层，即除N)
            // 而原始输入是整数，这里会显示浮点数。
            // 同时，我们使用的并行FFT是简化的，结果可能不正确。
            std::cout << global_result_ifft[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
} 