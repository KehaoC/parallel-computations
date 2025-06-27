#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Function to initialize matrices with random values
void init_matrices(float *A, float *B, int size) {
    for (int i = 0; i < size * size; i++) {
        A[i] = (float)rand() / RAND_MAX * 100.0f;
        B[i] = (float)rand() / RAND_MAX * 100.0f;
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

// Function to perform matrix multiplication with OpenMP
void matrix_multiply_omp(float *A, float *B, float *C, int size, int num_threads, const char *schedule) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <matrix_size> <num_threads> <schedule_type>\n", argv[0]);
        printf("Schedule types: default, static, dynamic\n");
        return 1;
    }

    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    const char *schedule = argv[3];

    // Validate input parameters
    if (size < 128 || size > 2048) {
        printf("Matrix size must be between 128 and 2048\n");
        return 1;
    }

    // Allocate memory for matrices
    float *A = (float *)malloc(size * size * sizeof(float));
    float *B = (float *)malloc(size * size * sizeof(float));
    float *C = (float *)malloc(size * size * sizeof(float));

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    srand(time(NULL));
    init_matrices(A, B, size);

    // Set OpenMP schedule
    if (strcmp(schedule, "static") == 0) {
        omp_set_schedule(omp_sched_static, 0);
    } else if (strcmp(schedule, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, 0);
    }

    // Measure computation time
    double start_time = omp_get_wtime();
    matrix_multiply_omp(A, B, C, size, num_threads, schedule);
    double end_time = omp_get_wtime();

    // Print results
    printf("Matrix A:\n");
    print_matrix(A, size);
    printf("Matrix B:\n");
    print_matrix(B, size);
    printf("Result Matrix C:\n");
    print_matrix(C, size);
    printf("Computation time: %.6f seconds\n", end_time - start_time);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
} 