#include "../include/implicit_parallel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

bool is_symmetric_implicit(float **matrix, int n, long double* time) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    #pragma GCC unroll 4
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        #pragma GCC unroll 4
        #pragma GCC ivdep
        for (int j = 0; j < i; j++) {
            if (matrix[i][j] != matrix[j][i]) {
                clock_gettime(CLOCK_MONOTONIC, &end);
                *time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
                printf("Computed that the matrix is not symmetric with implicit parallelization in: %Lf\n", *time);
                return false;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    *time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Computed that the matrix is symmetric with implicit parallelization in: %Lf\n", *time);

    return true;
}

float** transpose_implicit(float **matrix, int n, long double* time) {
    float **result = malloc(n * sizeof(float*));

    #pragma GCC unroll 4
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        result[i] = malloc(n * sizeof(float));
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    #pragma GCC unroll 4
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        #pragma GCC unroll 4
        #pragma GCC ivdep
        for (int j = 0; j < n; j++) {
            result[i][j] = matrix[j][i];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    *time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sequentially computed the transpose using implicit parallelization in: %Lf\n", *time);

    return result;
}
