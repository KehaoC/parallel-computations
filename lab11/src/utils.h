#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 错误检查宏
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 计时函数
inline double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 随机初始化数据
inline void random_init(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 10000) / 10000.0f;
    }
}

// 验证结果是否相同
inline bool verify_result(float *a, float *b, int size, float epsilon = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > epsilon) {
            printf("验证失败: a[%d] = %f, b[%d] = %f\n", i, a[i], i, b[i]);
            return false;
        }
    }
    return true;
}

// 打印部分结果用于调试
inline void print_partial_result(float *data, int height, int width, int channels, const char *name) {
    printf("%s (部分结果):\n", name);
    for (int c = 0; c < channels; c++) {
        printf("通道 %d:\n", c);
        for (int i = 0; i < min(5, height); i++) {
            for (int j = 0; j < min(5, width); j++) {
                printf("%.4f ", data[(c * height + i) * width + j]);
            }
            printf("...\n");
        }
        printf("...\n");
    }
}

#endif // UTILS_H
