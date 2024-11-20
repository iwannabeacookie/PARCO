#include "../include/sequential.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

bool is_symmetric(float **matrix, int n, double* time) {
    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        #pragma GCC unroll 4
        #pragma GCC ivdep
        for (int j = 0; j < i; j++) {
            if (matrix[i][j] != matrix[j][i]) {
                printf("Sequentially computed that the matrix is not symmetric in: %f\n", omp_get_wtime() - start);
                return false;
            }
        }
    }

    printf("Sequentially computed that the matrix is symmetric in: %f\n", omp_get_wtime() - start);

    return true;
}

bool is_symmetric_implicit(float **matrix, int n, double* time) {
    double start = omp_get_wtime();

    #pragma GCC unroll 4
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        #pragma GCC unroll 4
        #pragma GCC ivdep
        for (int j = 0; j < i; j++) {
            if (matrix[i][j] != matrix[j][i]) {
                printf("Computed that the matrix is not symmetric with implicit parallelization in: %f\n", omp_get_wtime() - start);
                return false;
            }
        }
    }

    printf("Computed that the matrix is symmetric with implicit parallelization in: %f\n", omp_get_wtime() - start);

    return true;
}

float** transpose(float **matrix, int n, double* time) {
    float **result = malloc(n * sizeof(float*));

    for (int i = 0; i < n; i++) {
        result[i] = malloc(n * sizeof(float));
    }

    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = matrix[j][i];
        }
    }

    printf("Sequentially computed the transpose in: %f\n", omp_get_wtime() - start);

    return result;
}

float** transpose_implicit(float **matrix, int n, double* time) {
    float **result = malloc(n * sizeof(float*));

    #pragma GCC unroll 4
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        result[i] = malloc(n * sizeof(float));
    }

    double start = omp_get_wtime();

    #pragma GCC unroll 4
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        #pragma GCC unroll 4
        #pragma GCC ivdep
        for (int j = 0; j < n; j++) {
            result[i][j] = matrix[j][i];
        }
    }

    printf("Sequentially computed the transpose using implicit parallelization in: %f\n", omp_get_wtime() - start);

    return result;
}
