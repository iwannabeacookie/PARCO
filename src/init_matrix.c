#include "../include/init_matrix.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

float** init_matrix_sequential(int n) {
    float start = omp_get_wtime();
    srand(time(NULL));

    float** matrix = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((float)(rand() % (int)10e6) / 1000);
        }
    }

    printf("Initialized matrix sequentialy in: %f\n", omp_get_wtime() - start);

    return matrix;
}

float** init_matrix_parallel(int n) {
    float start = omp_get_wtime();
    unsigned int seed = time(NULL);

    float** matrix = (float**)malloc(n * sizeof(float*));

    #pragma omp parallel for schedule(dynamic) private(seed)
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((float)(rand_r(&seed) % (int)10e6) / 1000);
        }
    }

    printf("Initialized matrix in parallel in: %f\n", omp_get_wtime() - start);

    return matrix;
}
