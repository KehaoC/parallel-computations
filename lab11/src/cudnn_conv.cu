#include "utils.h"
#include <cudnn.h>

// cuDNN错误检查宏
#define CHECK_CUDNN_ERROR(call) \
do { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN Error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudnnGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuDNN卷积实现
void cudnn_conv(float *h_input, float *h_kernel, float *h_output,
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
    
    // 创建cuDNN句柄
    cudnnHandle_t cudnn;
    CHECK_CUDNN_ERROR(cudnnCreate(&cudnn));
    
    // 创建张量描述符
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&kernel_desc));
    CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
    
    // 设置张量描述符
    // NCHW格式：批次大小(N)=1, 通道数(C)=channels, 高度(H)=in_height, 宽度(W)=in_width
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, channels, in_height, in_width));
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, channels, out_height, out_width));
    
    // 设置卷积核描述符
    // KCRS格式：输出通道数(K)=channels, 输入通道数(C)=channels, 高度(R)=kernel_size, 宽度(S)=kernel_size
    CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               channels, channels, kernel_size, kernel_size));
    
    // 设置卷积描述符
    CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride,
                                                    1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    
    // 获取最优算法
    cudnnConvolutionFwdAlgo_t algo;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    CHECK_CUDNN_ERROR(cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, kernel_desc, conv_desc, output_desc,
                                                        1, &returnedAlgoCount, &perfResults));
    algo = perfResults.algo;
    
    // 获取工作空间大小
    size_t workspace_size = 0;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, kernel_desc, conv_desc, output_desc,
                                                            algo, &workspace_size));
    
    // 分配工作空间
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&workspace, workspace_size));
    }
    
    // 执行卷积并计时
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cudaDeviceSynchronize();
    double start_time = get_time();
    
    CHECK_CUDNN_ERROR(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, kernel_desc, d_kernel,
                                            conv_desc, algo, workspace, workspace_size, &beta,
                                            output_desc, d_output));
    
    cudaDeviceSynchronize();
    double end_time = get_time();
    *time_taken = end_time - start_time;
    
    // 将结果从设备复制到主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // 清理资源
    if (workspace) {
        CHECK_CUDA_ERROR(cudaFree(workspace));
    }
    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(kernel_desc));
    CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN_ERROR(cudnnDestroy(cudnn));
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
    CHECK_CUDA_ERROR(cudaFree(d_output));
} 