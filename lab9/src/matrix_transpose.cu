#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For srand and timing
#include <cuda_runtime.h> // For cudaEvent_t
#include <math.h> // For fabs in verification

// Optional: Prints a small part of a matrix
void show_matrix_preview(float *mat, int R, int C, const char* name) {
    printf("\n%s (Max 10x10 Preview):\n", name);
    for (int i = 0; i < R && i < 10; ++i) {
        for (int j = 0; j < C && j < 10; ++j) {
            printf("%8.2f ", mat[i * C + j]);
        }
        printf("\n");
    }
}

// Kernel 1: Naive matrix transpose
__global__ void transpose_naive(float *in_data, float *out_data, int width, int height) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < height && c < width) {
        out_data[c * height + r] = in_data[r * width + c];
    }
}

#define TILE_S 32 // Tile size for shared memory version

// Kernel 2: Matrix transpose using shared memory
__global__ void transpose_shared_mem(float *in_data, float *out_data, int width, int height) {
    __shared__ float tile_cache[TILE_S][TILE_S]; 

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global index to load from input matrix
    int load_c = bx * TILE_S + tx;
    int load_r = by * TILE_S + ty;

    if (load_r < height && load_c < width) {
        tile_cache[ty][tx] = in_data[load_r * width + load_c];
    }

    __syncthreads(); 

    // Global index to store to output matrix (transposed logic)
    int store_c = by * TILE_S + tx; // Target column in output tile block
    int store_r = bx * TILE_S + ty; // Target row in output tile block

    if (store_r < width && store_c < height) { 
        out_data[store_r * height + store_c] = tile_cache[tx][ty]; // Read from tile using transposed thread indices
    }
}


int main(int argc, char **argv) {
    printf("--- CUDA Matrix Transpose ---\n");

    int N = 1024; 
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 32) N = 32; // Min sensible size for TILE_S
        if (N > 8192) N = 8192; // Practical upper limit
    }
    printf("Matrix Dim: %d x %d\n", N, N);

    int mat_w = N;
    int mat_h = N;
    size_t mat_bytes = (size_t)mat_w * mat_h * sizeof(float);

    float *h_in, *h_gpu_out, *h_cpu_out;
    h_in = (float*)malloc(mat_bytes);
    h_gpu_out = (float*)malloc(mat_bytes);
    h_cpu_out = (float*)malloc(mat_bytes);

    if (!h_in || !h_gpu_out || !h_cpu_out) {
        fprintf(stderr, "Host malloc failed for size %zu!\n", mat_bytes);
        return 1;
    }

    srand(time(NULL)); 
    for (int i = 0; i < mat_h * mat_w; ++i) {
        h_in[i] = (float)rand() / (float)RAND_MAX * 100.0f;
    }

    float *d_in = NULL, *d_out = NULL;
    cudaError_t cuda_stat;
    cuda_stat = cudaMalloc((void**)&d_in, mat_bytes);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "Dev malloc d_in: %s\n", cudaGetErrorString(cuda_stat)); goto end_prog; }
    cuda_stat = cudaMalloc((void**)&d_out, mat_bytes);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "Dev malloc d_out: %s\n", cudaGetErrorString(cuda_stat)); goto end_prog; }

    printf("Copying H->D...\n");
    cuda_stat = cudaMemcpy(d_in, h_in, mat_bytes, cudaMemcpyHostToDevice);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "Memcpy H2D: %s\n", cudaGetErrorString(cuda_stat)); goto end_prog; }

    cudaEvent_t timer_start, timer_stop;
    cudaEventCreate(&timer_start);
    cudaEventCreate(&timer_stop);
    float duration_ms = 0;

    // --- Naive --- 
    printf("\n--- Naive Transpose ---\n");
    dim3 block_cfg_naive(16, 16); 
    dim3 grid_cfg_naive((mat_w + block_cfg_naive.x - 1) / block_cfg_naive.x,
                        (mat_h + block_cfg_naive.y - 1) / block_cfg_naive.y);

    cudaEventRecord(timer_start, 0);
    transpose_naive<<<grid_cfg_naive, block_cfg_naive>>>(d_in, d_out, mat_w, mat_h);
    cudaEventRecord(timer_stop, 0);
    cudaEventSynchronize(timer_stop); 
    
    cuda_stat = cudaGetLastError(); 
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "Naive kernel fail: %s\n", cudaGetErrorString(cuda_stat)); goto events_done; }
    cudaEventElapsedTime(&duration_ms, timer_start, timer_stop);
    printf("Naive Kernel Time: %f ms\n", duration_ms);

    cuda_stat = cudaMemcpy(h_gpu_out, d_out, mat_bytes, cudaMemcpyDeviceToHost);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "Memcpy D2H (Naive): %s\n", cudaGetErrorString(cuda_stat)); goto events_done; }
    // show_matrix_preview(h_in, mat_h, mat_w, "Original");
    // show_matrix_preview(h_gpu_out, mat_w, mat_h, "GPU Naive Transposed"); 

    // --- Shared Memory --- 
    printf("\n--- Shared Memory Transpose ---\n");
    dim3 block_cfg_shared(TILE_S, TILE_S);
    dim3 grid_cfg_shared((mat_w + TILE_S - 1) / TILE_S,
                         (mat_h + TILE_S - 1) / TILE_S);

    cudaMemset(d_out, 0, mat_bytes); 
    cudaDeviceSynchronize(); 

    cudaEventRecord(timer_start, 0);
    transpose_shared_mem<<<grid_cfg_shared, block_cfg_shared>>>(d_in, d_out, mat_w, mat_h);
    cudaEventRecord(timer_stop, 0);
    cudaEventSynchronize(timer_stop);

    cuda_stat = cudaGetLastError(); 
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "SharedMem kernel fail: %s\n", cudaGetErrorString(cuda_stat)); goto events_done; }
    cudaEventElapsedTime(&duration_ms, timer_start, timer_stop);
    printf("Shared Mem Kernel Time: %f ms\n", duration_ms);

    cuda_stat = cudaMemcpy(h_gpu_out, d_out, mat_bytes, cudaMemcpyDeviceToHost);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "Memcpy D2H (Shared): %s\n", cudaGetErrorString(cuda_stat)); goto events_done; }
    // show_matrix_preview(h_gpu_out, mat_w, mat_h, "GPU SharedMem Transposed");

    // --- CPU Verification ---
    printf("\nCPU transpose for verification...\n");
    clock_t cpu_clk_start = clock();
    for (int r = 0; r < mat_h; ++r) {
        for (int c = 0; c < mat_w; ++c) {
            h_cpu_out[c * mat_h + r] = h_in[r * mat_w + c];
        }
    }
    clock_t cpu_clk_end = clock();
    double cpu_ms = ((double)(cpu_clk_end - cpu_clk_start)) * 1000.0 / CLOCKS_PER_SEC; 
    printf("CPU Transpose Time: %f ms\n", cpu_ms);

    int mismatches = 0;
    double max_err = 0.0;
    for (int i = 0; i < mat_w * mat_h; ++i) { 
        double cur_err = fabs((double)h_gpu_out[i] - (double)h_cpu_out[i]);
        if (cur_err > 1e-4) { 
            mismatches++;
            if (cur_err > max_err) max_err = cur_err;
        }
    }
    if (mismatches == 0) {
        printf("\nVerification: PASS (GPU result matches CPU)\n");
    } else {
        printf("\nVerification: FAIL (%d mismatches, Max Error: %e)\n", mismatches, max_err);
    }

events_done:
    cudaEventDestroy(timer_start);
    cudaEventDestroy(timer_stop);
end_prog:
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (h_in) free(h_in);
    if (h_gpu_out) free(h_gpu_out);
    if (h_cpu_out) free(h_cpu_out);

    printf("\n--- End ---\n");
    return 0;
} 