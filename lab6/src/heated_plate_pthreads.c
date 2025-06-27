#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>   // For timing, using struct timespec and clock_gettime
#include <math.h>   // For fabs
#include "parallel_for.h"

#define N 100 // Grid size (N x N)
#define MAX_ITER 1000 // Maximum iterations
#define TOLERANCE 1e-4 // Tolerance for convergence

// Global variables for the grid (accessible by functor)
double **grid_w;    // Current iteration grid
double **grid_w_new; // Next iteration grid

// Structure to pass arguments to the functor for parallel_for
typedef struct {
    // Potentially other arguments if needed, like N, but N is global via #define
    // For this specific problem, the index 'i' from parallel_for is enough.
} heated_plate_args;

// Functor for the parallel_for loop
void *heated_plate_functor(int i, void *arg_bundle) {
    // heated_plate_args *args = (heated_plate_args *)arg_bundle; // Cast if you have specific args
    // No specific args needed from arg_bundle for this simple case, as i is the row index.

    for (int j = 1; j < N - 1; j++) {
        // Using the formula: w_new[i][j] = 0.25 * (w[i-1][j-1] + w[i-1][j+1] + w[i+1][j-1] + w[i+1][j+1])
        grid_w_new[i][j] = 0.25 * (grid_w[i-1][j-1] + grid_w[i-1][j+1] + grid_w[i+1][j-1] + grid_w[i+1][j+1]);
    }
    return NULL;
}

// Function to initialize the grid
void initialize_grid(double **current_grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                current_grid[i][j] = 0.0; // Boundary conditions
            } else if (i == N / 4 && j == N / 4) {
                current_grid[i][j] = 100.0; // Hot spot 1
            } else if (i == 3 * N / 4 && j == 3 * N / 4) {
                current_grid[i][j] = 100.0; // Hot spot 2
            } else {
                current_grid[i][j] = 50.0; // Initial temperature
            }
        }
    }
}

// Function to print the grid (for debugging)
void print_grid(double **current_grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", current_grid[i][j]);
        }
        printf("\n");
    }
}

// Helper for timing
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
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

    // Allocate memory for grids
    grid_w = (double **)malloc(N * sizeof(double *));
    grid_w_new = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        grid_w[i] = (double *)malloc(N * sizeof(double));
        grid_w_new[i] = (double *)malloc(N * sizeof(double));
    }

    initialize_grid(grid_w);

    // Copy initial w to w_new to maintain boundary conditions for the first iteration's w_new
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid_w_new[i][j] = grid_w[i][j];
        }
    }

    heated_plate_args p_args; // No actual members, but pass it for parallel_for signature

    double start_time = get_time_sec();
    double diff = TOLERANCE + 1.0;
    int iter = 0;

    while (iter < MAX_ITER && diff >= TOLERANCE) {
        // The parallel_for will iterate over rows i from 1 to N-2 (inclusive for N-1 exclusive end)
        // The functor will handle the inner loop over j
        parallel_for(1, N - 1, 1, heated_plate_functor, &p_args, num_threads);

        // Calculate overall difference and swap grids (sequentially)
        double max_cell_diff = 0.0;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                double cell_diff = fabs(grid_w_new[i][j] - grid_w[i][j]);
                if (cell_diff > max_cell_diff) {
                    max_cell_diff = cell_diff;
                }
                grid_w[i][j] = grid_w_new[i][j]; // Update current grid
            }
        }
        diff = max_cell_diff;
        
        // Note: Boundary conditions in grid_w_new are not updated by the parallel_for or the copy above.
        // If they could change or need to be strictly maintained in w_new as well, they should be copied.
        // For this problem, w_new's boundaries are effectively ignored as only interior points are computed and copied back.

        iter++;
        if (iter % 100 == 0) {
            // printf("Iteration: %d, Max Difference: %f\n", iter, diff);
        }
    }

    double end_time = get_time_sec();

    printf("Pthreads Heated Plate Simulation\n");
    printf("Grid Size: %d x %d\n", N, N);
    printf("Max Iterations: %d\n", MAX_ITER);
    printf("Tolerance: %e\n", TOLERANCE);
    printf("Threads: %d\n", num_threads);
    printf("Converged after %d iterations.\n", iter);
    printf("Maximum difference: %f\n", diff);
    printf("Total execution time: %f seconds\n", end_time - start_time);

    // Optional: Print final grid
    // if (N <= 20) {
    //     printf("\nFinal Grid:\n");
    //     print_grid(grid_w);
    // }

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(grid_w[i]);
        free(grid_w_new[i]);
    }
    free(grid_w);
    free(grid_w_new);

    return 0;
} 