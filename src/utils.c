#include <stdio.h>
#include <time.h>
#include "../include/utils.h"
#include "../include/sequential.h"
#include "../include/parallel.h"
#include "../include/config.h"

void print_matrix(float** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void correct_transpose(float** m1, float** m2, int n) {
    int equal = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            equal = m1[i][j] == m2[i][j];
            if (!equal) {
                printf("Not correct\n");
                return;
            }
        }
    }
    printf("Correct\n");
}

// Don't use that function on bigger inputs
void test_randomness(float** m1, float** m2, int n) {
    int occurrences = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                for (int t = 0; t < n; t++) {
                    if (m1[i][j] == m2[k][t]) {
                        occurrences++;
                    }
                    if (occurrences > 1) {
                        printf("Not random\n");
                        return;
                    }
                }
            }
            occurrences = 0;
        }
    }
    printf("Random\n");
}

double get_time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void benchmark_function(void (*func)(float**, int, double*), float** matrix, int n, const char* func_name) {
    double total_time = 0.0;
    for (int i = 0; i < NUM_RUNS; i++) {
        double time;
        func(matrix, n, &time);
        total_time += time;
    }
    printf("%s average time: %f seconds\n", func_name, total_time / NUM_RUNS);
}

void is_symmetric_wrapper(float** matrix, int n, double* time) {
    is_symmetric(matrix, n, time);
}

void is_symmetric_implicit_wrapper(float** matrix, int n, double* time) {
    is_symmetric_implicit(matrix, n, time);
}

void is_symmetric_omp_wrapper(float** matrix, int n, double* time) {
    is_symmetric_omp(matrix, n, time);
}

void transpose_wrapper(float** matrix, int n, double* time) {
    transpose(matrix, n, time);
}

void transpose_omp_wrapper(float** matrix, int n, double* time) {
    transpose_omp(matrix, n, time);
}

void transpose_omp_block_based_wrapper(float** matrix, int n, double* time) {
    transpose_omp_block_based(matrix, n, BLOCK_SIZE, time); // Assuming block size of 64
}
