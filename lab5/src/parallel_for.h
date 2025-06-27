#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#include <pthread.h>

// Function pointer type for the work to be done in parallel
typedef void *(*parallel_for_functor)(int, void*);

// Structure to hold thread arguments
typedef struct {
    int start;
    int end;
    int increment;
    parallel_for_functor functor;
    void *args;
} thread_args;

// Main parallel_for function
void parallel_for(int start, int end, int increment, 
                 parallel_for_functor functor, void *args, 
                 int num_threads);

#endif // PARALLEL_FOR_H 