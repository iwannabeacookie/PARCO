#define BLOCK_SIZE 16

#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

bool is_symmetric_omp(float **matrix, int n) {
    double start = omp_get_wtime();
    bool is_symmetric = true;

    #pragma omp parallel shared(is_symmetric) 
    {
        // Using reduction significantly improves performance
        #pragma omp for collapse(2) reduction(&&:is_symmetric)
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

    printf("Computed that the matrix is %ssymmetric using OMP in: %f\n", is_symmetric ? "" : "not ", omp_get_wtime() - start);

    return is_symmetric;
}

float** transpose_omp(float **matrix, int n) {
    double start = omp_get_wtime();

    float **result = malloc(n * sizeof(float*));

    #pragma omp parallel
    {
        // Adding static schedule improves performance sometimes, but adds a lot of variance to execution time
        // Using dynamic schedule grants better performance and is more consistent
        // Using guided schedule seems to have the greatest overall boost, but the consistency is even worse, than static
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            result[i] = malloc(n * sizeof(float));
        }

        // Imperically deduced that including a collapse clause is faster
        // using schedule(dynamic) is slower than using schedule(static)
        // using schedule(guided) is the fastest
        #pragma omp for collapse(2) schedule(guided)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = matrix[j][i];
            }
        }
    }

    printf("Computed the transpose using OMP in: %f\n", omp_get_wtime() - start);

    return result;
}

float** transpose_omp_block_based(float **matrix, int n) {
    double start = omp_get_wtime();

    float **result = malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        result[i] = malloc(n * sizeof(float));
    }

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < n; i += BLOCK_SIZE) {
            for (int j = 0; j < n; j += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < n; jj++) {
                        result[ii][jj] = matrix[jj][ii];
                    }
                }
            }
        }
    }

    printf("Computed the block-based transpose using OMP in: %f\n", omp_get_wtime() - start);

    return result;
}
