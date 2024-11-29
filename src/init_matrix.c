#include "../include/init_matrix.h"
#include "../include/config.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

float** init_matrix_sequential(int n) {
    double start = omp_get_wtime();
    srand(time(NULL));

    float** matrix = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((float)(rand() % (int)10e6) / 1000);
        }
    }

    if (get_config()->VERBOSE_LEVEL > 1) {
        printf("Matrix generated sequentialy:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("Initialized matrix sequentialy in: %f\n", omp_get_wtime() - start);
    }

    return matrix;
}

// Before optimizitations sequential approach was faster
float** init_matrix_parallel(int n) {
    double start = omp_get_wtime();

    float** matrix = (float**)malloc(n * sizeof(float*));

    // Turns out, using a private seed makes each thread produce the same randomized numbers (who would have thought?)
    //
    // Sharing the seed variable doesn't seem to affect the randomness
    // UPD: it does :(
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) + omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            matrix[i] = (float*)malloc(n * sizeof(float));
            for (int j = 0; j < n; j++) {
                matrix[i][j] = ((float)(rand_r(&seed) % (int)10e6) / 1000);
            }
        }
    }

    if (get_config()->VERBOSE_LEVEL > 1) {
        printf("Matrix generated in parallel:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("Initialized matrix in parallel in: %f\n", omp_get_wtime() - start);
    }

    return matrix;
}
