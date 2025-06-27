#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to initialize matrix with random values
void init_matrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 10.0;  // Random values between 0 and 10
    }
}

// Function to print matrix
void print_matrix(double* matrix, int rows, int cols, const char* name) {
    printf("\nMatrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: mpirun -np <num_processes> ./matrix_multiply <matrix_size> <matrix_size> <matrix_size>\n");
        return 1;
    }

    int matrix_size = atoi(argv[1]);
    if (matrix_size < 128 || matrix_size > 2048) {
        printf("Matrix size must be between 128 and 2048\n");
        return 1;
    }

    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Calculate rows per process
    int rows_per_process = matrix_size / world_size;

    // Allocate memory for local matrices
    double* local_A = (double*)malloc(rows_per_process * matrix_size * sizeof(double));
    double* local_B = (double*)malloc(matrix_size * matrix_size * sizeof(double));
    double* local_C = (double*)malloc(rows_per_process * matrix_size * sizeof(double));

    // Initialize matrices on root process
    if (world_rank == 0) {
        srand(time(NULL));
        double* A = (double*)malloc(matrix_size * matrix_size * sizeof(double));
        double* B = (double*)malloc(matrix_size * matrix_size * sizeof(double));
        
        init_matrix(A, matrix_size, matrix_size);
        init_matrix(B, matrix_size, matrix_size);

        // Scatter matrix A to all processes
        MPI_Scatter(A, rows_per_process * matrix_size, MPI_DOUBLE,
                   local_A, rows_per_process * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Broadcast matrix B to all processes
        MPI_Bcast(B, matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        memcpy(local_B, B, matrix_size * matrix_size * sizeof(double));

        free(A);
        free(B);
    } else {
        // Receive scattered data for non-root processes
        MPI_Scatter(NULL, rows_per_process * matrix_size, MPI_DOUBLE,
                   local_A, rows_per_process * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(local_B, matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Start timing
    double start_time = MPI_Wtime();

    // Perform local matrix multiplication
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < matrix_size; j++) {
            double sum = 0.0;
            for (int k = 0; k < matrix_size; k++) {
                sum += local_A[i * matrix_size + k] * local_B[k * matrix_size + j];
            }
            local_C[i * matrix_size + j] = sum;
        }
    }

    // Gather results to root process
    double* C = NULL;
    if (world_rank == 0) {
        C = (double*)malloc(matrix_size * matrix_size * sizeof(double));
    }

    MPI_Gather(local_C, rows_per_process * matrix_size, MPI_DOUBLE,
               C, rows_per_process * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // End timing
    double end_time = MPI_Wtime();

    // Print results on root process
    if (world_rank == 0) {
        printf("\nMatrix multiplication completed with %d processes\n", world_size);
        printf("Matrix size: %d x %d\n", matrix_size, matrix_size);
        printf("Time taken: %.6f seconds\n", end_time - start_time);
        
        // Print first few elements of result matrix
        printf("\nFirst 4x4 elements of result matrix:\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%.2f ", C[i * matrix_size + j]);
            }
            printf("\n");
        }
        
        free(C);
    }

    // Cleanup
    free(local_A);
    free(local_B);
    free(local_C);

    MPI_Finalize();
    return 0;
} 