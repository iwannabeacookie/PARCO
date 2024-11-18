#include "../include/sequential.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

bool is_symmetric(float **matrix, int n) {
    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
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

float** transpose(float **matrix, int n) {
    double start = omp_get_wtime();

    float **result = malloc(n * sizeof(float*));

    for (int i = 0; i < n; i++) {
        result[i] = malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            result[i][j] = matrix[j][i];
        }
    }

    printf("Sequentially computed the transpose in: %f\n", omp_get_wtime() - start);

    return result;
}

float** transpose_implicit(float **matrix, int n) {
    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j+=4) {
            matrix[i][j] = matrix[j][i];
            matrix[i][j+1] = matrix[j+1][i];
            matrix[i][j+2] = matrix[j+2][i];
            matrix[i][j+3] = matrix[j+3][i];
        }
    }

    printf("Sequentially computed the transpose using implicit parallelization in: %f\n", omp_get_wtime() - start);

    return matrix;
}
