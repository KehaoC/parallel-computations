// lab10/src/gemm.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "utils.h"

// 包含其他实现
#include "gemm_naive.cu"
#include "gemm_shared.cu"

int main(int argc, char *argv[]) {
    // 设置随机种子
    srand(time(NULL));
    
    // 检查命令行参数
    if (argc != 4) {
        printf("用法: %s <M> <K> <N>\n", argv[0]);
        printf("其中 M, K, N 是矩阵维度（范围：128-2048）\n");
        return 1;
    }
    
    // 解析命令行参数
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    // 验证参数范围
    if (M < 128 || M > 2048 || K < 128 || K > 2048 || N < 128 || N > 2048) {
        printf("错误: 矩阵维度必须在128到2048之间\n");
        return 1;
    }
    
    printf("矩阵乘法: (%d x %d) * (%d x %d) = (%d x %d)\n", M, K, K, N, M, N);
    
    // 分配主机内存
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    
    // 初始化输入矩阵
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);
    
    // 打印输入矩阵（仅部分）
    printMatrix("矩阵 A", A, M, K);
    printMatrix("矩阵 B", B, K, N);
    
    // 测试不同的线程块大小
    int blockSizes[] = {8, 16, 32};
    int numBlockSizes = sizeof(blockSizes) / sizeof(int);
    
    printf("\n朴素CUDA矩阵乘法：\n");
    printf("线程块大小\t执行时间(ms)\t验证结果\n");
    
    for (int i = 0; i < numBlockSizes; i++) {
        int blockSize = blockSizes[i];
        dim3 block(blockSize, blockSize);
        
        // 执行朴素版本
        float time = matrixMultiplyNaive(A, B, C, M, N, K, block);
        
        // 验证结果
        bool correct = verifyResult(A, B, C, M, N, K);
        
        printf("%dx%d\t\t%.4f\t\t%s\n", blockSize, blockSize, time, correct ? "通过" : "失败");
    }
    
    printf("\n共享内存优化的CUDA矩阵乘法：\n");
    printf("线程块大小\t执行时间(ms)\t验证结果\n");
    
    // 测试共享内存版本（使用模板特化）
    float time8 = matrixMultiplyShared<8>(A, B, C, M, N, K);
    bool correct8 = verifyResult(A, B, C, M, N, K);
    printf("8x8\t\t%.4f\t\t%s\n", time8, correct8 ? "通过" : "失败");
    
    float time16 = matrixMultiplyShared<16>(A, B, C, M, N, K);
    bool correct16 = verifyResult(A, B, C, M, N, K);
    printf("16x16\t\t%.4f\t\t%s\n", time16, correct16 ? "通过" : "失败");
    
    float time32 = matrixMultiplyShared<32>(A, B, C, M, N, K);
    bool correct32 = verifyResult(A, B, C, M, N, K);
    printf("32x32\t\t%.4f\t\t%s\n", time32, correct32 ? "通过" : "失败");
    
    // 打印输出矩阵（仅部分）
    printMatrix("结果矩阵 C", C, M, N);
    
    // 释放主机内存
    free(A);
    free(B);
    free(C);
    
    return 0;
}