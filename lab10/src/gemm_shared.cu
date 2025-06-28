// lab10/src/gemm_shared.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

// 使用共享内存的CUDA矩阵乘法核函数
template <int BLOCK_SIZE>
__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K) {
    // 分配共享内存
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    // 计算当前线程对应的行和列
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有需要的分块
    for (int i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        // 加载数据到共享内存
        if (row < M && i * BLOCK_SIZE + tx < K) {
            sharedA[ty][tx] = A[row * K + i * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (i * BLOCK_SIZE + ty < K && col < N) {
            sharedB[ty][tx] = B[(i * BLOCK_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        // 同步以确保所有数据都已加载
        __syncthreads();
        
        // 计算当前分块的部分和
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        // 同步以确保所有线程都完成了计算
        __syncthreads();
    }
    
    // 将结果写回全局内存
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 使用共享内存的矩阵乘法包装函数
template <int BLOCK_SIZE>
float matrixMultiplyShared(float *A, float *B, float *C, int M, int N, int K) {
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    
    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 计算网格大小
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 创建CUDA事件来测量执行时间
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // 记录开始时间
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // 启动核函数
    matrixMulShared<BLOCK_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    
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