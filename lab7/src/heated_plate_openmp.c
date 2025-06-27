#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h> // For timing

// #define N 100 // Grid size (N x N) - Will be a variable
#define MAX_ITER 1000 // Maximum iterations
#define TOLERANCE 1e-4 // Tolerance for convergence

// Function to initialize the grid
void initialize_grid(double **grid, int current_N) {
    for (int i = 0; i < current_N; i++) {
        for (int j = 0; j < current_N; j++) {
            if (i == 0 || i == current_N - 1 || j == 0 || j == current_N - 1) {
                grid[i][j] = 0.0; // Boundary conditions (e.g., cool edges)
            } else if (i == current_N / 4 && j == current_N / 4) {
                grid[i][j] = 100.0; // Hot spot 1
            } else if (i == 3 * current_N / 4 && j == 3 * current_N / 4) {
                grid[i][j] = 100.0; // Hot spot 2
            }
            else {
                grid[i][j] = 50.0; // Initial temperature for interior points
            }
        }
    }
}

// Function to print the grid (for debugging)
void print_grid(double **grid, int current_N) {
    for (int i = 0; i < current_N; i++) {
        for (int j = 0; j < current_N; j++) {
            printf("%.2f ", grid[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int N_param = 100; // Default Grid size
    int num_threads = 1; // Default number of threads

    if (argc > 1) {
        N_param = atoi(argv[1]);
        if (N_param <= 1) { // N must be at least 2 for sensible interior points
            printf("Grid size N must be greater than 1. Using default N=100.\n");
            N_param = 100;
        }
    }
    if (argc > 2) {
        num_threads = atoi(argv[2]);
        if (num_threads <= 0) {
            printf("Number of threads must be positive. Using 1 thread.\n");
            num_threads = 1;
        }
    }

    omp_set_num_threads(num_threads);

    double **w = (double **)malloc(N_param * sizeof(double *));
    double **w_new = (double **)malloc(N_param * sizeof(double *));
    for (int i = 0; i < N_param; i++) {
        w[i] = (double *)malloc(N_param * sizeof(double));
        w_new[i] = (double *)malloc(N_param * sizeof(double));
    }

    initialize_grid(w, N_param);

    // Copy initial w to w_new to maintain boundary conditions for the first iteration's w_new
    for (int i = 0; i < N_param; i++) {
        for (int j = 0; j < N_param; j++) {
            w_new[i][j] = w[i][j];
        }
    }

    double start_time = omp_get_wtime();
    double diff = TOLERANCE + 1.0; // Ensure the loop runs at least once
    int iter = 0;

    while (iter < MAX_ITER && diff >= TOLERANCE) {
        // diff = 0.0; // Reset before parallel region if reduction is used for max

        // Parallelize the main computation loop
        #pragma omp parallel for collapse(2) // Removed reduction(+:diff) as diff calculation is sequential below
        for (int i = 1; i < N_param - 1; i++) {
            for (int j = 1; j < N_param - 1; j++) {
                w_new[i][j] = 0.25 * (w[i-1][j-1] + w[i-1][j+1] + w[i+1][j-1] + w[i+1][j+1]);
            }
        }
        
        double max_cell_diff = 0.0;
        for (int i = 1; i < N_param - 1; i++) {
            for (int j = 1; j < N_param - 1; j++) {
                double cell_diff = w_new[i][j] - w[i][j];
                if (cell_diff < 0) cell_diff = -cell_diff;
                if (cell_diff > max_cell_diff) {
                    max_cell_diff = cell_diff;
                }
                w[i][j] = w_new[i][j];
            }
        }
        diff = max_cell_diff;

        iter++;
//        if (iter % 100 == 0) {
//            printf("Iteration: %d, Max Difference: %f\n", iter, diff);
//        }
    }

    double end_time = omp_get_wtime();

    printf("OpenMP Heated Plate Simulation\n");
    printf("Grid Size (N_param): %d x %d\n", N_param, N_param);
    printf("Max Iterations: %d\n", MAX_ITER);
    printf("Tolerance: %e\n", TOLERANCE);
    printf("Threads: %d\n", num_threads);
    printf("Converged after %d iterations.\n", iter);
    printf("Maximum difference: %f\n", diff);
    printf("Total execution time: %f seconds\n", end_time - start_time);

    // Optional: Print final grid (can be very large)
//    if (N_param <= 20) { // Only print for small grids
//        printf("\nFinal Grid:\n");
//        print_grid(w, N_param);
//    }

    // Free allocated memory
    for (int i = 0; i < N_param; i++) {
        free(w[i]);
        free(w_new[i]);
    }
    free(w);
    free(w_new);

    return 0;
} 