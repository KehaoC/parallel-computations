#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "parallel_for.h"

#define BLOCK_SIZE 32

// Structure to hold matrix multiplication arguments
typedef struct {
    float *A;
    float *B;
    float *C;
    int size;
    int block_size;
} matrix_args;

// Functor for matrix multiplication
void *matrix_mult_functor(int idx, void *args) {
    matrix_args *margs = (matrix_args *)args;
    int size = margs->size;
    int block_size = margs->block_size;
    
    // Calculate block coordinates
    int block_row = (idx / (size / block_size)) * block_size;
    int block_col = (idx % (size / block_size)) * block_size;
    
    // Process block
    for (int i = block_row; i < block_row + block_size && i < size; i++) {
        for (int j = block_col; j < block_col + block_size && j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += margs->A[i * size + k] * margs->B[k * size + j];
            }
            margs->C[i * size + j] = sum;
        }
    }
    
    return NULL;
}

// Function to initialize matrices with random values
void init_matrices(float *A, float *B, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = (float)rand() / RAND_MAX * 100.0f;
            B[i * size + j] = (float)rand() / RAND_MAX * 100.0f;
        }
    }
}

// Function to print a matrix
void print_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.2f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <num_threads>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    // Validate input parameters
    if (size < 128 || size > 2048) {
        printf("Matrix size must be between 128 and 2048\n");
        return 1;
    }

    // Allocate memory for matrices with padding for better cache alignment
    size_t matrix_size = size * size * sizeof(float);
    float *A = (float *)aligned_alloc(64, matrix_size);
    float *B = (float *)aligned_alloc(64, matrix_size);
    float *C = (float *)aligned_alloc(64, matrix_size);

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    srand(time(NULL));
    init_matrices(A, B, size);

    // Prepare arguments for parallel_for
    matrix_args args = {
        .A = A,
        .B = B,
        .C = C,
        .size = size,
        .block_size = BLOCK_SIZE
    };

    // Calculate number of blocks
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_blocks = num_blocks * num_blocks;

    // Measure computation time
    clock_t start_time = clock();
    parallel_for(0, total_blocks, 1, matrix_mult_functor, &args, num_threads);
    clock_t end_time = clock();

    // Print results
    printf("Matrix A:\n");
    print_matrix(A, size);
    printf("Matrix B:\n");
    print_matrix(B, size);
    printf("Result Matrix C:\n");
    print_matrix(C, size);
    printf("Computation time: %.6f seconds\n", 
           (double)(end_time - start_time) / CLOCKS_PER_SEC);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
} 