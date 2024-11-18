#include "../include/init_matrix.h"
#include <time.h>
#include <stdlib.h>
#include <omp.h>

float** init_matrix_sequential(int n) {
    srand(time(NULL));

    float** matrix = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((float)(rand() % (int)10e6) / 1000);
        }
    }

    return matrix;
}

float** init_matrix_parallel(int n) {
    srand(time(NULL));

    float** matrix = (float**)malloc(n * sizeof(float*));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 100;
        }
    }

    return matrix;
}
