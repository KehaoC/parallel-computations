#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "parallel_for.h"

// Thread worker function
void *thread_worker(void *arg) {
    thread_args *args = (thread_args *)arg;
    
    for (int i = args->start; i < args->end; i += args->increment) {
        args->functor(i, args->args);
    }
    
    return NULL;
}

// Main parallel_for implementation
void parallel_for(int start, int end, int increment, 
                 parallel_for_functor functor, void *args, 
                 int num_threads) {
    if (num_threads <= 0) {
        printf("Error: Number of threads must be positive\n");
        return;
    }

    // Create thread array and thread arguments array
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    thread_args *thread_args_array = (thread_args *)malloc(num_threads * sizeof(thread_args));

    if (!threads || !thread_args_array) {
        printf("Error: Memory allocation failed\n");
        free(threads);
        free(thread_args_array);
        return;
    }

    // Calculate chunk size for each thread
    int total_iterations = (end - start + increment - 1) / increment;
    int chunk_size = (total_iterations + num_threads - 1) / num_threads;

    // Create and start threads
    for (int i = 0; i < num_threads; i++) {
        int thread_start = start + i * chunk_size * increment;
        int thread_end = thread_start + chunk_size * increment;
        if (thread_end > end) thread_end = end;

        thread_args_array[i].start = thread_start;
        thread_args_array[i].end = thread_end;
        thread_args_array[i].increment = increment;
        thread_args_array[i].functor = functor;
        thread_args_array[i].args = args;

        if (pthread_create(&threads[i], NULL, thread_worker, &thread_args_array[i]) != 0) {
            printf("Error: Failed to create thread %d\n", i);
            // Clean up already created threads
            for (int j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(thread_args_array);
            return;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Clean up
    free(threads);
    free(thread_args_array);
} 