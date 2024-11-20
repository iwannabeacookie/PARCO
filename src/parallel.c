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
        // #pragma omp for collapse(2) reduction(&&:is_symmetric)
        // Had to get rid of the collapse clause because it is not fully supported with the older OMP versions on hpcc
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

    printf("Computed that the matrix is %ssymmetric using OMP in: %f\n", is_symmetric ? "" : "not ", omp_get_wtime() - start);

    return is_symmetric;
}

float** transpose_omp(float **matrix, int n) {
    double start;

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

        #pragma omp single
        {
            start = omp_get_wtime();
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

float** transpose_omp_block_based(float **matrix, int n, int block_size) {
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

    printf("Computed the block-based (size %d) transpose using OMP in: %f\n", block_size, omp_get_wtime() - start);

    return result;
}
