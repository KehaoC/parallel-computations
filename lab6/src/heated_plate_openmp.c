#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h> // For timing

#define N 100 // Grid size (N x N)
#define MAX_ITER 1000 // Maximum iterations
#define TOLERANCE 1e-4 // Tolerance for convergence

// Function to initialize the grid
void initialize_grid(double **grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                grid[i][j] = 0.0; // Boundary conditions (e.g., cool edges)
            } else if (i == N / 4 && j == N / 4) {
                grid[i][j] = 100.0; // Hot spot 1
            } else if (i == 3 * N / 4 && j == 3 * N / 4) {
                grid[i][j] = 100.0; // Hot spot 2
            }
            else {
                grid[i][j] = 50.0; // Initial temperature for interior points
            }
        }
    }
}

// Function to print the grid (for debugging)
void print_grid(double **grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", grid[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int num_threads = 1;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0) {
            printf("Number of threads must be positive. Using 1 thread.\n");
            num_threads = 1;
        }
    }

    omp_set_num_threads(num_threads);

    double **w = (double **)malloc(N * sizeof(double *));
    double **w_new = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        w[i] = (double *)malloc(N * sizeof(double));
        w_new[i] = (double *)malloc(N * sizeof(double));
    }

    initialize_grid(w);

    // Copy initial w to w_new to maintain boundary conditions for the first iteration's w_new
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            w_new[i][j] = w[i][j];
        }
    }

    double start_time = omp_get_wtime();
    double diff = TOLERANCE + 1.0; // Ensure the loop runs at least once
    int iter = 0;

    while (iter < MAX_ITER && diff >= TOLERANCE) {
        diff = 0.0;

        // Parallelize the main computation loop
        #pragma omp parallel for collapse(2) reduction(+:diff)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                // Using the formula: w_new[i][j] = 0.25 * (w[i-1][j-1] + w[i-1][j+1] + w[i+1][j-1] + w[i+1][j+1])
                w_new[i][j] = 0.25 * (w[i-1][j-1] + w[i-1][j+1] + w[i+1][j-1] + w[i+1][j+1]);
                double current_diff = w_new[i][j] - w[i][j];
                if (current_diff < 0) current_diff = -current_diff; // abs
                if (current_diff > diff) { // Note: This reduction for max diff is not perfectly thread-safe without critical or atomic.
                                          // However, for heated plate, sum of absolute differences is more common for convergence.
                                          // Let's use sum of absolute differences for simplicity and better parallel reduction.
                                          // Reverted to a simpler approach for now, will refine if needed.
                                          // For now, we calculate diff sequentially after the parallel loop.
                }
            }
        }
        
        // Calculate overall difference and swap grids (sequentially for now to avoid complex reduction)
        // This can be parallelized too, but let's keep it simple for the base OpenMP version.
        double max_cell_diff = 0.0;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
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
        if (iter % 100 == 0) {
            // printf("Iteration: %d, Max Difference: %f\n", iter, diff);
        }
    }

    double end_time = omp_get_wtime();

    printf("OpenMP Heated Plate Simulation\n");
    printf("Grid Size: %d x %d\n", N, N);
    printf("Max Iterations: %d\n", MAX_ITER);
    printf("Tolerance: %e\n", TOLERANCE);
    printf("Threads: %d\n", num_threads);
    printf("Converged after %d iterations.\n", iter);
    printf("Maximum difference: %f\n", diff);
    printf("Total execution time: %f seconds\n", end_time - start_time);

    // Optional: Print final grid (can be very large)
    // if (N <= 20) { // Only print for small grids
    //     printf("\nFinal Grid:\n");
    //     print_grid(w);
    // }

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(w[i]);
        free(w_new[i]);
    }
    free(w);
    free(w_new);

    return 0;
} 