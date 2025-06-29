#include "utils.h"

// im2col核函数：将输入图像重排为列矩阵
__global__ void im2col_kernel(float *input, float *output,
                             int in_height, int in_width, int out_height, int out_width,
                             int kernel_size, int stride, int padding, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < out_height * out_width) {
        int out_x = idx % out_width;
        int out_y = idx / out_width;
        
        // 对于每个输出位置，提取对应的输入patch
        for (int c = 0; c < channels; c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = out_y * stride - padding + ky;
                    int in_x = out_x * stride - padding + kx;
                    
                    // 输出索引计算
                    int out_idx = ((c * kernel_size * kernel_size + ky * kernel_size + kx) * out_height * out_width) + (out_y * out_width + out_x);
                    
                    // 检查边界并设置值
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        output[out_idx] = input[(c * in_height + in_y) * in_width + in_x];
                    } else {
                        output[out_idx] = 0.0f;
                    }
                }
            }
        }
    }
}

// 矩阵乘法核函数
__global__ void gemm_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// im2col的CPU实现（用于验证）
void im2col_cpu(float *input, float *output,
               int in_height, int in_width, int out_height, int out_width,
               int kernel_size, int stride, int padding, int channels) {
    for (int out_y = 0; out_y < out_height; out_y++) {
        for (int out_x = 0; out_x < out_width; out_x++) {
            for (int c = 0; c < channels; c++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_y = out_y * stride - padding + ky;
                        int in_x = out_x * stride - padding + kx;
                        
                        // 输出索引计算
                        int out_idx = ((c * kernel_size * kernel_size + ky * kernel_size + kx) * out_height * out_width) + (out_y * out_width + out_x);
                        
                        // 检查边界并设置值
                        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            output[out_idx] = input[(c * in_height + in_y) * in_width + in_x];
                        } else {
                            output[out_idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// 矩阵乘法的CPU实现（用于验证）
void gemm_cpu(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// im2col + GEMM卷积的GPU实现接口
void im2col_conv_gpu(float *h_input, float *h_kernel, float *h_output,
                    int in_height, int in_width, int kernel_size, int stride, int padding, int channels,
                    double *time_taken) {
    // 计算输出尺寸
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // 计算内存大小
    size_t input_size = channels * in_height * in_width * sizeof(float);
    size_t kernel_size_bytes = channels * channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = channels * out_height * out_width * sizeof(float);
    
    // im2col后的矩阵大小
    int col_height = channels * kernel_size * kernel_size;
    int col_width = out_height * out_width;
    size_t col_size = col_height * col_width * sizeof(float);
    
    // 重排卷积核为行矩阵
    int kernel_row = channels;  // 输出通道数
    int kernel_col = channels * kernel_size * kernel_size;  // 输入通道数 * 卷积核大小
    size_t kernel_matrix_size = kernel_row * kernel_col * sizeof(float);
    
    // 分配设备内存
    float *d_input, *d_kernel, *d_output, *d_col, *d_kernel_matrix;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_col, col_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel_matrix, kernel_matrix_size));
    
    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice));
    
    // 同步设备并开始计时
    cudaDeviceSynchronize();
    double start_time = get_time();
    
    // 步骤1: 执行im2col操作
    int im2col_threads = 256;
    int im2col_blocks = (out_height * out_width + im2col_threads - 1) / im2col_threads;
    im2col_kernel<<<im2col_blocks, im2col_threads>>>(d_input, d_col, 
                                                   in_height, in_width, out_height, out_width,
                                                   kernel_size, stride, padding, channels);
    
    // 步骤2: 重排卷积核
    // 这一步可以在主机上完成，然后传输到设备
    float *h_kernel_matrix = (float *)malloc(kernel_matrix_size);
    for (int out_c = 0; out_c < channels; out_c++) {
        for (int c = 0; c < channels; c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int kernel_idx = ((out_c * channels + c) * kernel_size + ky) * kernel_size + kx;
                    int matrix_idx = out_c * kernel_col + (c * kernel_size * kernel_size + ky * kernel_size + kx);
                    h_kernel_matrix[matrix_idx] = h_kernel[kernel_idx];
                }
            }
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel_matrix, h_kernel_matrix, kernel_matrix_size, cudaMemcpyHostToDevice));
    free(h_kernel_matrix);
    
    // 步骤3: 执行GEMM操作
    // M = 输出通道数, N = 输出高度*宽度, K = 输入通道数 * 卷积核大小
    int M = channels;
    int N = out_height * out_width;
    int K = channels * kernel_size * kernel_size;
    
    dim3 gemm_block(16, 16);
    dim3 gemm_grid((N + gemm_block.x - 1) / gemm_block.x, (M + gemm_block.y - 1) / gemm_block.y);
    gemm_kernel<<<gemm_grid, gemm_block>>>(d_kernel_matrix, d_col, d_output, M, N, K);
    
    // 检查核函数执行错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 同步设备并结束计时
    cudaDeviceSynchronize();
    double end_time = get_time();
    *time_taken = end_time - start_time;
    
    // 将结果从设备复制到主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_col));
    CHECK_CUDA_ERROR(cudaFree(d_kernel_matrix));
} 