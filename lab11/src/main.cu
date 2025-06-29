#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 声明三种卷积方法的接口
extern void direct_conv_gpu(float *h_input, float *h_kernel, float *h_output,
                          int in_height, int in_width, int kernel_size, int stride, int padding, int channels,
                          double *time_taken);

extern void direct_conv_cpu(float *input, float *kernel, float *output,
                          int in_height, int in_width, int out_height, int out_width,
                          int kernel_size, int stride, int padding, int channels);

extern void im2col_conv_gpu(float *h_input, float *h_kernel, float *h_output,
                          int in_height, int in_width, int kernel_size, int stride, int padding, int channels,
                          double *time_taken);

extern void cudnn_conv(float *h_input, float *h_kernel, float *h_output,
                     int in_height, int in_width, int kernel_size, int stride, int padding, int channels,
                     double *time_taken);

// 运行单次测试
void run_test(int in_size, int kernel_size, int stride, int padding, int channels) {
    printf("\n============================================================\n");
    printf("测试参数: 输入大小=%dx%d, 卷积核大小=%dx%d, 步幅=%d, 填充=%d, 通道数=%d\n", 
           in_size, in_size, kernel_size, kernel_size, stride, padding, channels);
    
    // 计算输出尺寸
    int out_size = (in_size + 2 * padding - kernel_size) / stride + 1;
    printf("输出大小: %dx%d\n", out_size, out_size);
    
    // 分配内存
    size_t input_size = channels * in_size * in_size * sizeof(float);
    size_t kernel_size_bytes = channels * channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = channels * out_size * out_size * sizeof(float);
    
    float *h_input = (float *)malloc(input_size);
    float *h_kernel = (float *)malloc(kernel_size_bytes);
    float *h_output_direct = (float *)malloc(output_size);
    float *h_output_im2col = (float *)malloc(output_size);
    float *h_output_cudnn = (float *)malloc(output_size);
    float *h_output_cpu = (float *)malloc(output_size);
    
    // 初始化输入和卷积核
    srand(42);  // 固定随机种子以便结果可重现
    random_init(h_input, channels * in_size * in_size);
    random_init(h_kernel, channels * channels * kernel_size * kernel_size);
    
    // 执行CPU卷积（用于验证）
    if (in_size <= 64) {  // 仅对小尺寸输入执行CPU验证
        printf("执行CPU卷积（用于验证）...\n");
        direct_conv_cpu(h_input, h_kernel, h_output_cpu, in_size, in_size, out_size, out_size, 
                      kernel_size, stride, padding, channels);
    }
    
    // 执行直接卷积
    double direct_time = 0.0;
    printf("执行直接卷积（滑窗法）...\n");
    direct_conv_gpu(h_input, h_kernel, h_output_direct, in_size, in_size, 
                  kernel_size, stride, padding, channels, &direct_time);
    printf("直接卷积耗时: %.6f 秒\n", direct_time);
    
    // 执行im2col + GEMM卷积
    double im2col_time = 0.0;
    printf("执行im2col + GEMM卷积...\n");
    im2col_conv_gpu(h_input, h_kernel, h_output_im2col, in_size, in_size, 
                  kernel_size, stride, padding, channels, &im2col_time);
    printf("im2col + GEMM卷积耗时: %.6f 秒\n", im2col_time);
    
    // 执行cuDNN卷积
    double cudnn_time = 0.0;
    printf("执行cuDNN卷积...\n");
    cudnn_conv(h_input, h_kernel, h_output_cudnn, in_size, in_size, 
             kernel_size, stride, padding, channels, &cudnn_time);
    printf("cuDNN卷积耗时: %.6f 秒\n", cudnn_time);
    
    // 验证结果
    if (in_size <= 64) {  // 仅对小尺寸输入进行验证
        printf("\n验证结果:\n");
        bool direct_correct = verify_result(h_output_direct, h_output_cpu, channels * out_size * out_size);
        printf("直接卷积结果正确: %s\n", direct_correct ? "是" : "否");
        
        bool im2col_correct = verify_result(h_output_im2col, h_output_cpu, channels * out_size * out_size);
        printf("im2col + GEMM卷积结果正确: %s\n", im2col_correct ? "是" : "否");
        
        bool cudnn_correct = verify_result(h_output_cudnn, h_output_cpu, channels * out_size * out_size);
        printf("cuDNN卷积结果正确: %s\n", cudnn_correct ? "是" : "否");
    }
    
    // 性能对比
    printf("\n性能对比:\n");
    printf("直接卷积 vs im2col + GEMM: %.2f倍\n", direct_time / im2col_time);
    printf("直接卷积 vs cuDNN: %.2f倍\n", direct_time / cudnn_time);
    printf("im2col + GEMM vs cuDNN: %.2f倍\n", im2col_time / cudnn_time);
    
    // 释放内存
    free(h_input);
    free(h_kernel);
    free(h_output_direct);
    free(h_output_im2col);
    free(h_output_cudnn);
    free(h_output_cpu);
}

int main(int argc, char **argv) {
    // 默认参数
    int input_size = 32;
    int kernel_size = 3;
    
    // 解析命令行参数
    if (argc >= 3) {
        input_size = atoi(argv[1]);
        kernel_size = atoi(argv[2]);
    } else {
        printf("用法: %s <输入大小> <卷积核大小>\n", argv[0]);
        printf("使用默认参数: 输入大小=%d, 卷积核大小=%d\n", input_size, kernel_size);
    }
    
    // 固定参数
    int channels = 3;
    int padding = 1;  // 使用填充以保持输出尺寸
    
    // 测试不同步幅
    for (int stride = 1; stride <= 3; stride++) {
        run_test(input_size, kernel_size, stride, padding, channels);
    }
    
    // 如果输入尺寸较小，测试更大的输入尺寸
    if (input_size <= 64) {
        int larger_sizes[] = {64, 128, 256, 512};
        for (int i = 0; i < 4; i++) {
            if (larger_sizes[i] > input_size) {
                for (int stride = 1; stride <= 3; stride++) {
                    run_test(larger_sizes[i], kernel_size, stride, padding, channels);
                }
            }
        }
    }
    
    return 0;
} 