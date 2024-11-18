#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

bool is_symmetric_omp(float **matrix, int n) {
    double start = omp_get_wtime();
    bool is_symmetric = true;

    #pragma omp parallel shared(is_symmetric)
    {
        #pragma omp for collapse(2)
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

    if (is_symmetric) {
        printf("Computed that the matrix is symmetric using OMP in: %f\n", omp_get_wtime() - start);
    } else {
        printf("Computed that the matrix is not symmetric using OMP in: %f\n", omp_get_wtime() - start);
    }

    return is_symmetric;
}

float** transpose_omp(float **matrix, int n) {
    double start = omp_get_wtime();

    float **result = malloc(n * sizeof(float*));

    #pragma omp parallel
    {
        #pragma omp for 
        for (int i = 0; i < n; i++) {
            result[i] = malloc(n * sizeof(float));
        }

        #pragma omp for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = matrix[j][i];
            }
        }
    }

    printf("Parallelly computed the transpose in: %f\n", omp_get_wtime() - start);

    return result;
}
