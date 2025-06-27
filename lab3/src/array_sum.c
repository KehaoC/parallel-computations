#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

// 线程参数结构体
typedef struct {
    int thread_id;
    int num_threads;
    double* array;
    long long size;
    double partial_sum;
} ThreadArgs;

// 初始化数组
void init_array(double* array, long long size) {
    for (long long i = 0; i < size; i++) {
        array[i] = (double)rand() / RAND_MAX;
    }
}

// 串行数组求和
double serial_array_sum(double* array, long long size) {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double sum = 0.0;
    for (long long i = 0; i < size; i++) {
        sum += array[i];
    }

    gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Serial time: %.6f seconds\n", time_taken);
    return sum;
}

// 并行数组求和的线程函数
void* array_sum_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    long long start_idx = (args->thread_id * args->size) / args->num_threads;
    long long end_idx = ((args->thread_id + 1) * args->size) / args->num_threads;

    args->partial_sum = 0.0;
    for (long long i = start_idx; i < end_idx; i++) {
        args->partial_sum += args->array[i];
    }

    return NULL;
}

// 并行数组求和
double parallel_array_sum(double* array, long long size, int num_threads) {
    struct timeval start, end;
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* thread_args = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));

    gettimeofday(&start, NULL);

    for (int i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_threads;
        thread_args[i].array = array;
        thread_args[i].size = size;
        thread_args[i].partial_sum = 0.0;
        pthread_create(&threads[i], NULL, array_sum_thread, &thread_args[i]);
    }

    double total_sum = 0.0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_sum += thread_args[i].partial_sum;
    }

    gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Parallel time (%d threads): %.6f seconds\n", num_threads, time_taken);

    free(threads);
    free(thread_args);

    return total_sum;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <array_size> <num_threads>\n", argv[0]);
        return 1;
    }

    long long size = atoll(argv[1]);
    int num_threads = atoi(argv[2]);
    srand(time(NULL));

    // 分配并初始化数组
    double* array = (double*)malloc(size * sizeof(double));
    init_array(array, size);

    // 执行串行求和
    double serial_sum = serial_array_sum(array, size);
    printf("Serial sum: %.6f\n", serial_sum);

    // 执行并行求和
    double parallel_sum = parallel_array_sum(array, size, num_threads);
    printf("Parallel sum: %.6f\n", parallel_sum);
    printf("Difference: %.10f\n", serial_sum - parallel_sum);

    // 释放内存
    free(array);

    return 0;
} 