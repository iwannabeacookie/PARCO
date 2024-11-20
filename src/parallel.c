#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

bool is_symmetric_omp(float **matrix, int n, double* time) {
    double start = omp_get_wtime();
    bool is_symmetric = true;

    #pragma omp parallel shared(is_symmetric) 
    {
        #pragma omp for reduction(&&:is_symmetric)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (matrix[i][j] != matrix[j][i]) {
                    #pragma omp atomic write
                    is_symmetric = false;
                    #pragma omp cancel for
                }
                #pragma omp cancellation point for
            }
        }
    }

    *time = omp_get_wtime() - start;
    printf("Computed that the matrix is %ssymmetric using OMP in: %f\n", is_symmetric ? "" : "not ", *time);

    return is_symmetric;
}

float** transpose_omp(float **matrix, int n, double* time) {
    double start;

    float **result = malloc(n * sizeof(float*));

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            result[i] = malloc(n * sizeof(float));
        }

        #pragma omp single
        {
            start = omp_get_wtime();
        }

        #pragma omp for collapse(2) schedule(guided)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = matrix[j][i];
            }
        }
    }

    *time = omp_get_wtime() - start;
    printf("Computed the transpose using OMP in: %f\n", *time);

    return result;
}

float** transpose_omp_block_based(float **matrix, int n, int block_size, double* time) {
    double start;

    float **result = malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        result[i] = malloc(n * sizeof(float));
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            start = omp_get_wtime();
        }

        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < n; i += block_size) {
            for (int j = 0; j < n; j += block_size) {
                for (int ii = i; ii < i + block_size && ii < n; ii++) {
                    for (int jj = j; jj < j + block_size && jj < n; jj++) {
                        result[ii][jj] = matrix[jj][ii];
                    }
                }
            }
        }
    }

    *time = omp_get_wtime() - start;
    printf("Computed the block-based (size %d) transpose using OMP in: %f\n", block_size, *time);

    return result;
}
