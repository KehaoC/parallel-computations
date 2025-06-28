// lab10/src/utils.h
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 初始化矩阵（随机值）
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(rand()) / RAND_MAX;
    }
}

// 打印矩阵（用于调试，只打印一小部分）
void printMatrix(const char* name, float *matrix, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    
    int max_print = 6; // 最多打印6x6的子矩阵
    int r_print = (rows < max_print) ? rows : max_print;
    int c_print = (cols < max_print) ? cols : max_print;
    
    for (int i = 0; i < r_print; i++) {
        for (int j = 0; j < c_print; j++) {
            printf("%.4f ", matrix[i * cols + j]);
        }
        if (c_print < cols) printf("...");
        printf("\n");
    }
    if (r_print < rows) printf("...\n");
    printf("\n");
}

// 验证矩阵乘法结果
bool verifyResult(float *A, float *B, float *C, int M, int N, int K) {
    float *C_host = (float*)malloc(M * N * sizeof(float));
    
    // 计算CPU版本的矩阵乘法结果
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_host[i * N + j] = sum;
        }
    }
    
    // 验证结果
    const float epsilon = 1e-4;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C_host[i * N + j] - C[i * N + j]) > epsilon) {
                printf("Verification failed at [%d, %d]: CPU=%f, GPU=%f, diff=%e\n", 
                       i, j, C_host[i * N + j], C[i * N + j], 
                       fabs(C_host[i * N + j] - C[i * N + j]));
                free(C_host);
                return false;
            }
        }
    }
    
    free(C_host);
    return true;
}

#endif // UTILS_H