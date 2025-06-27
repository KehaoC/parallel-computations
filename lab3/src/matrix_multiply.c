#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

// 矩阵结构体
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

// 线程参数结构体
typedef struct {
    int thread_id;
    int num_threads;
    Matrix* A;
    Matrix* B;
    Matrix* C;
} ThreadArgs;

// 创建矩阵
Matrix* create_matrix(int rows, int cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat->data[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// 初始化矩阵
void init_matrix(Matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// 释放矩阵内存
void free_matrix(Matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
    free(mat);
}

// 并行矩阵乘法的线程函数
void* matrix_multiply_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int start_row = (args->thread_id * args->A->rows) / args->num_threads;
    int end_row = ((args->thread_id + 1) * args->A->rows) / args->num_threads;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < args->B->cols; j++) {
            args->C->data[i][j] = 0.0;
            for (int k = 0; k < args->A->cols; k++) {
                args->C->data[i][j] += args->A->data[i][k] * args->B->data[k][j];
            }
        }
    }
    return NULL;
}

// 串行矩阵乘法
double serial_matrix_multiply(Matrix* A, Matrix* B, Matrix* C) {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            C->data[i][j] = 0.0;
            for (int k = 0; k < A->cols; k++) {
                C->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }

    gettimeofday(&end, NULL);
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

// 并行矩阵乘法
double parallel_matrix_multiply(Matrix* A, Matrix* B, Matrix* C, int num_threads) {
    struct timeval start, end;
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* thread_args = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));

    gettimeofday(&start, NULL);

    for (int i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_threads;
        thread_args[i].A = A;
        thread_args[i].B = B;
        thread_args[i].C = C;
        pthread_create(&threads[i], NULL, matrix_multiply_thread, &thread_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end, NULL);

    free(threads);
    free(thread_args);

    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <num_threads>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    srand(time(NULL));

    // 创建并初始化矩阵
    Matrix* A = create_matrix(size, size);
    Matrix* B = create_matrix(size, size);
    Matrix* C_serial = create_matrix(size, size);
    Matrix* C_parallel = create_matrix(size, size);

    init_matrix(A);
    init_matrix(B);

    // 执行串行乘法
    double serial_time = serial_matrix_multiply(A, B, C_serial);
    printf("Serial time: %.6f seconds\n", serial_time);

    // 执行并行乘法
    double parallel_time = parallel_matrix_multiply(A, B, C_parallel, num_threads);
    printf("Parallel time (%d threads): %.6f seconds\n", num_threads, parallel_time);
    printf("Speedup: %.2fx\n", serial_time / parallel_time);

    // 释放内存
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_serial);
    free_matrix(C_parallel);

    return 0;
} 