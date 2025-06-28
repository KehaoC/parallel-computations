// lab10/src/gemm_naive.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

// 朴素的CUDA矩阵乘法核函数
__global__ void matrixMulNaive(float *A, float *B, float *C, int M, int N, int K) {
    // 计算当前线程对应的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查是否越界
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 朴素矩阵乘法的包装函数
float matrixMultiplyNaive(float *A, float *B, float *C, int M, int N, int K, dim3 blockSize) {
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    
    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 计算网格大小
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    // 创建CUDA事件来测量执行时间
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // 记录开始时间
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // 启动核函数
    matrixMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    
    // 检查核函数执行是否有错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 记录结束时间
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    // 计算执行时间
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // 将结果从设备复制到主机
    CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 释放设备内存和事件
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return milliseconds;
}