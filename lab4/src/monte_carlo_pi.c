#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

typedef struct {
    long long points_per_thread;
    long long points_in_circle;
    int thread_id;
    unsigned int seed;
} ThreadData;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
long long total_points_in_circle = 0;

void* calculate_pi(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    long long local_points_in_circle = 0;
    
    for (long long i = 0; i < data->points_per_thread; i++) {
        double x = (double)rand_r(&data->seed) / RAND_MAX;
        double y = (double)rand_r(&data->seed) / RAND_MAX;
        
        if (x*x + y*y <= 1.0) {
            local_points_in_circle++;
        }
    }
    
    data->points_in_circle = local_points_in_circle;
    
    pthread_mutex_lock(&mutex);
    total_points_in_circle += local_points_in_circle;
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <num_threads> <points>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    long long total_points = atoll(argv[2]);
    long long points_per_thread = total_points / num_threads;

    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = malloc(num_threads * sizeof(ThreadData));

    // Record start time
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].points_per_thread = points_per_thread;
        thread_data[i].thread_id = i;
        thread_data[i].seed = time(NULL) + i;  // Different seed for each thread
        pthread_create(&threads[i], NULL, calculate_pi, &thread_data[i]);
    }

    // Wait for all threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Record end time
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calculate pi
    double pi_estimate = 4.0 * total_points_in_circle / (double)(points_per_thread * num_threads);
    double error = fabs(pi_estimate - M_PI);
    
    // Calculate execution time
    double execution_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // Print results
    printf("Threads: %d\n", num_threads);
    printf("Total points: %lld\n", total_points);
    printf("Points in circle: %lld\n", total_points_in_circle);
    printf("Pi estimate: %.10f\n", pi_estimate);
    printf("Absolute error: %.10f\n", error);
    printf("Execution time: %.6f seconds\n", execution_time);

    // Cleanup
    free(threads);
    free(thread_data);
    pthread_mutex_destroy(&mutex);

    return 0;
} 