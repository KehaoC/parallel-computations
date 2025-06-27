#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

// 矩阵数据结构
typedef struct {
    int rows;
    int cols;
    double* data;
} Matrix;

// 创建自定义MPI数据类型
void create_matrix_type(Matrix* mat, MPI_Datatype* matrix_type) {
    int block_lengths[] = {1, 1, mat->rows * mat->cols};
    MPI_Aint displacements[3];
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_DOUBLE};
    
    // 计算偏移量
    MPI_Get_address(&mat->rows, &displacements[0]);
    MPI_Get_address(&mat->cols, &displacements[1]);
    MPI_Get_address(mat->data, &displacements[2]);
    
    // 使偏移量相对于基地址
    MPI_Aint base;
    MPI_Get_address(mat, &base);
    for (int i = 0; i < 3; i++) {
        displacements[i] = MPI_Aint_diff(displacements[i], base);
    }
    
    // 创建结构类型
    MPI_Type_create_struct(3, block_lengths, displacements, types, matrix_type);
    MPI_Type_commit(matrix_type);
}

// 初始化矩阵
void init_matrix(Matrix* mat, int rows, int cols) {
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double*)malloc(rows * cols * sizeof(double));
    
    for (int i = 0; i < rows * cols; i++) {
        mat->data[i] = (double)rand() / RAND_MAX;
    }
}

// 释放矩阵内存
void free_matrix(Matrix* mat) {
    free(mat->data);
}

// 矩阵乘法核心计算
void multiply_block(const double* A, const double* B, double* C,
                   int A_rows, int A_cols, int B_cols,
                   int start_row, int num_rows) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A_cols; k++) {
                sum += A[i * A_cols + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 设置随机数种子
    srand(time(NULL) + rank);
    
    // 矩阵大小（可以通过命令行参数传入）
    int matrix_size = 1024;  // 默认值
    if (argc > 1) {
        matrix_size = atoi(argv[1]);
    }
    
    // 计算每个进程处理的行数
    int rows_per_proc = matrix_size / size;
    int remainder = matrix_size % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // 分配内存
    double *B = (double*)malloc(matrix_size * matrix_size * sizeof(double));
    double *local_A = (double*)malloc(local_rows * matrix_size * sizeof(double));
    double *local_C = (double*)malloc(local_rows * matrix_size * sizeof(double));
    
    if (!B || !local_A || !local_C) {
        fprintf(stderr, "Memory allocation failed on process %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 在主进程上初始化矩阵
    if (rank == 0) {
        // 初始化矩阵A和B
        double *A = (double*)malloc(matrix_size * matrix_size * sizeof(double));
        if (!A) {
            fprintf(stderr, "Memory allocation failed for matrix A\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        for (int i = 0; i < matrix_size * matrix_size; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
        
        // 计算发送计数和位移
        int *sendcounts = (int*)malloc(size * sizeof(int));
        int *displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (matrix_size / size + (i < remainder ? 1 : 0)) * matrix_size;
            displs[i] = offset * matrix_size;
            offset += matrix_size / size + (i < remainder ? 1 : 0);
        }
        
        // 分发矩阵A
        MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                     local_A, local_rows * matrix_size, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        
        free(A);
        free(sendcounts);
        free(displs);
    } else {
        // 非主进程只接收它们的部分
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                     local_A, local_rows * matrix_size, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }
    
    // 广播矩阵B到所有进程
    MPI_Bcast(B, matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 执行本地矩阵乘法
    double start_time = MPI_Wtime();
    multiply_block(local_A, B, local_C, local_rows, matrix_size, matrix_size, 0, local_rows);
    double end_time = MPI_Wtime();
    
    // 收集结果
    double *C = NULL;
    if (rank == 0) {
        C = (double*)malloc(matrix_size * matrix_size * sizeof(double));
        if (!C) {
            fprintf(stderr, "Memory allocation failed for result matrix\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // 计算接收计数和位移
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        
        for (int i = 0; i < size; i++) {
            recvcounts[i] = (matrix_size / size + (i < remainder ? 1 : 0)) * matrix_size;
            displs[i] = offset * matrix_size;
            offset += matrix_size / size + (i < remainder ? 1 : 0);
        }
    }
    
    MPI_Gatherv(local_C, local_rows * matrix_size, MPI_DOUBLE,
                C, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // 输出计算时间
    double local_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Matrix size: %d x %d\n", matrix_size, matrix_size);
        printf("Number of processes: %d\n", size);
        printf("Computation time: %f seconds\n", max_time);
        
        free(C);
        free(recvcounts);
        free(displs);
    }
    
    // 清理内存
    free(local_A);
    free(local_C);
    free(B);
    
    MPI_Finalize();
    return 0;
} 