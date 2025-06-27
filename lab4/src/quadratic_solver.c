#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

typedef struct {
    double a;
    double b;
    double c;
    double x1;
    double x2;
    int thread_id;
} QuadraticParams;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int completed_threads = 0;

void* solve_quadratic(void* arg) {
    QuadraticParams* params = (QuadraticParams*)arg;
    double a = params->a;
    double b = params->b;
    double c = params->c;
    
    // Calculate discriminant
    double discriminant = b * b - 4 * a * c;
    
    // Calculate roots
    if (discriminant >= 0) {
        params->x1 = (-b + sqrt(discriminant)) / (2 * a);
        params->x2 = (-b - sqrt(discriminant)) / (2 * a);
    } else {
        params->x1 = params->x2 = NAN;
    }

    // Signal completion
    pthread_mutex_lock(&mutex);
    completed_threads++;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    QuadraticParams* params = malloc(num_threads * sizeof(QuadraticParams));

    // Initialize random seed
    srand(time(NULL));

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        params[i].thread_id = i;
        // Generate random coefficients
        params[i].a = (double)rand() / RAND_MAX * 10;
        params[i].b = (double)rand() / RAND_MAX * 10;
        params[i].c = (double)rand() / RAND_MAX * 10;

        pthread_create(&threads[i], NULL, solve_quadratic, &params[i]);
    }

    // Wait for all threads to complete
    pthread_mutex_lock(&mutex);
    while (completed_threads < num_threads) {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);

    // Print results
    for (int i = 0; i < num_threads; i++) {
        printf("Thread %d: a=%.2f, b=%.2f, c=%.2f\n", i, params[i].a, params[i].b, params[i].c);
        if (isnan(params[i].x1)) {
            printf("  No real roots\n");
        } else {
            printf("  x1=%.2f, x2=%.2f\n", params[i].x1, params[i].x2);
        }
    }

    // Cleanup
    free(threads);
    free(params);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
} 