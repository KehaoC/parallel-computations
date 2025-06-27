#include <stdio.h>

// CUDA Kernel: Each thread prints its ID and block ID
__global__ void жители_этажей(int total_blocks, int threads_x, int threads_y) {
    int b_id = blockIdx.x;      // Current block ID
    int t_id_x = threadIdx.x;   // Thread ID in x-dimension
    int t_id_y = threadIdx.y;   // Thread ID in y-dimension

    // Output format: Hello World from Thread (x, y) in Block z!
    printf("Hello World from Thread (%d, %d) in Block %d!\n", t_id_x, t_id_y, b_id);
}

int main() {
    printf("--- CUDA Hello World ---\n");

    int num_blocks = 2; // Number of thread blocks
    int block_dim_x = 2; // Threads per block in x-dim
    int block_dim_y = 2; // Threads per block in y-dim

    printf("Config: Blocks=%d, Threads/Block=%dx%d\n", num_blocks, block_dim_x, block_dim_y);

    dim3 grid_config(num_blocks);       // Grid dimension
    dim3 block_config(block_dim_x, block_dim_y); // Block dimension

    // Launch kernel
    жители_этажей<<<grid_config, block_config>>>(num_blocks, block_dim_x, block_dim_y);

    // Synchronize device
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(sync_err));
        return 1;
    }

    printf("Hello World from the host!\n");

    // The user is expected to observe the output order and note its non-deterministic nature.

    return 0;
} 