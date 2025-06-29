#include "utils.h"

// 直接卷积的CUDA核函数
__global__ void direct_conv_kernel(float *input, float *kernel, float *output,
                                  int in_height, int in_width, int out_height, int out_width,
                                  int kernel_size, int stride, int padding, int channels) {
    // 计算当前线程负责的输出位置
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    
    // 检查是否在输出范围内
    if (out_x < out_width && out_y < out_height && out_c < channels) {
        float sum = 0.0f;
        
        // 遍历所有输入通道
        for (int c = 0; c < channels; c++) {
            // 遍历卷积核
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    // 计算对应的输入位置
                    int in_y = out_y * stride - padding + ky;
                    int in_x = out_x * stride - padding + kx;
                    
                    // 检查是否在输入范围内
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        float in_val = input[(c * in_height + in_y) * in_width + in_x];
                        float kernel_val = kernel[((out_c * channels + c) * kernel_size + ky) * kernel_size + kx];
                        sum += in_val * kernel_val;
                    }
                }
            }
        }
        
        // 写入输出
        output[(out_c * out_height + out_y) * out_width + out_x] = sum;
    }
}

// 直接卷积的CPU实现（用于验证）
void direct_conv_cpu(float *input, float *kernel, float *output,
                    int in_height, int in_width, int out_height, int out_width,
                    int kernel_size, int stride, int padding, int channels) {
    // 对每个输出位置进行计算
    for (int out_c = 0; out_c < channels; out_c++) {
        for (int out_y = 0; out_y < out_height; out_y++) {
            for (int out_x = 0; out_x < out_width; out_x++) {
                float sum = 0.0f;
                
                // 遍历所有输入通道
                for (int c = 0; c < channels; c++) {
                    // 遍历卷积核
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            // 计算对应的输入位置
                            int in_y = out_y * stride - padding + ky;
                            int in_x = out_x * stride - padding + kx;
                            
                            // 检查是否在输入范围内
                            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                                float in_val = input[(c * in_height + in_y) * in_width + in_x];
                                float kernel_val = kernel[((out_c * channels + c) * kernel_size + ky) * kernel_size + kx];
                                sum += in_val * kernel_val;
                            }
                        }
                    }
                }
                
                // 写入输出
                output[(out_c * out_height + out_y) * out_width + out_x] = sum;
            }
        }
    }
}

// 直接卷积的GPU实现接口
void direct_conv_gpu(float *h_input, float *h_kernel, float *h_output,
                    int in_height, int in_width, int kernel_size, int stride, int padding, int channels,
                    double *time_taken) {
    // 计算输出尺寸
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // 计算内存大小
    size_t input_size = channels * in_height * in_width * sizeof(float);
    size_t kernel_size_bytes = channels * channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = channels * out_height * out_width * sizeof(float);
    
    // 分配设备内存
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    
    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice));
    
    // 设置线程块和网格大小
    dim3 block_size(16, 16);
    dim3 grid_size((out_width + block_size.x - 1) / block_size.x,
                  (out_height + block_size.y - 1) / block_size.y,
                  channels);
    
    // 同步设备并开始计时
    cudaDeviceSynchronize();
    double start_time = get_time();
    
    // 启动核函数
    direct_conv_kernel<<<grid_size, block_size>>>(d_input, d_kernel, d_output,
                                                in_height, in_width, out_height, out_width,
                                                kernel_size, stride, padding, channels);
    
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
} 