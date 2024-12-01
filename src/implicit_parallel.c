#include "../include/implicit_parallel.h"
#include "../include/config.h"
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
                
                if (get_config()->VERBOSE_LEVEL > 1) {
                    printf("Computed that the matrix is not symmetric with implicit parallelization in: %Lf\n", *time);
                }

                return false;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    *time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (get_config()->VERBOSE_LEVEL > 1) {
        printf("Computed that the matrix is symmetric with implicit parallelization in: %Lf\n", *time);
    }

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

    if (get_config()->VERBOSE_LEVEL > 1) {
        printf("Computed the transpose with implicit parallelization in: %Lf\n", *time);
    }

    return result;
}

float** transpose_implicit_block_based(float **matrix, int n, long double* time) {
    float **result = malloc(n * sizeof(float*));

    #pragma GCC unroll 4
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        result[i] = malloc(n * sizeof(float));
    }

    Config* config = get_config();
    int BLOCK_SIZE = config->BLOCK_SIZE;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    #pragma GCC unroll 4
    #pragma GCC ivdep
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        #pragma GCC unroll 4
        #pragma GCC ivdep
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            #pragma GCC unroll 4
            #pragma GCC ivdep
            for (int k = i; k < i + BLOCK_SIZE && k < n; k++) {
                #pragma GCC unroll 4
                #pragma GCC ivdep
                for (int l = j; l < j + BLOCK_SIZE && l < n; l++) {
                    result[k][l] = matrix[l][k];
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    *time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (get_config()->VERBOSE_LEVEL > 1) {
        printf("Computed the transpose with implicit parallelization in: %Lf\n", *time);
    }

    return result;
}

void transpose_implicit_recursive(float** original, float** transposed, int start_row, int start_col, int size, int n) {
    Config* cfg = get_config();
    if (size <= cfg->BLOCK_SIZE) {
        #pragma GCC unroll 4
        #pragma GCC ivdep
        for (int i = start_row; i < start_row + size; i++) {
            #pragma GCC unroll 4
            #pragma GCC ivdep
            for (int j = start_col; j < start_col + size; j++) {
                transposed[j][i] = original[i][j];
            }
        }
    } else {
        // Recursive case: divide the matrix into quadrants
        int half_size = size / 2;

        transpose_implicit_recursive(original, transposed, start_row, start_col, half_size, n);
        transpose_implicit_recursive(original, transposed, start_row, start_col + half_size, half_size, n);
        transpose_implicit_recursive(original, transposed, start_row + half_size, start_col, half_size, n);
        transpose_implicit_recursive(original, transposed, start_row + half_size, start_col + half_size, half_size, n);
    }
}

float** transpose_implicit_cache_oblivious(float ** matrix, int n, long double* time) {
    float **result = malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        result[i] = malloc(n * sizeof(float));
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    transpose_implicit_recursive(matrix, result, 0, 0, n, n);

    clock_gettime(CLOCK_MONOTONIC, &end);
    *time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (get_config()->VERBOSE_LEVEL > 1) {
        printf("Computed the transpose with implicit parallelization in: %Lf\n", *time);
    }

    return result;
}
