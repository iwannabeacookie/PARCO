#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "../include/utils.h"
#include "../include/sequential.h"
#include "../include/omp_parallel.h"
#include "../include/config.h"
#include "../include/implicit_parallel.h"

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

                if (get_config()->VERBOSE_LEVEL > 0) {
                    printf("Transpose is not correct\n");
                }
                return;
            }
        }
    }

    if (get_config()->VERBOSE_LEVEL > 0) {
        printf("Transpose is correct\n");
    }
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

void print_loading_bar(int progress, int total) {
    int bar_width = 50;
    float progress_ratio = (float)progress / total;
    int pos = bar_width * progress_ratio;

    printf("[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d%%\r", (int)(progress_ratio * 100));
    fflush(stdout);
}

void deallocate_matrix(float** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void benchmark_function(void (*func)(float**, int, long double*), float** matrix, int n, const char* func_name) {
    Config* cfg = get_config();
    double total_time = 0.0;
    for (int i = 0; i < cfg->NUM_RUNS; i++) {
        print_loading_bar(i, cfg->NUM_RUNS);
        long double time;
        func(matrix, n, &time);
        total_time += time;
    }

    printf("\x1b[2K");

    if (cfg->VERBOSE_LEVEL > 0) {
        printf("%s average time: %f seconds\n", func_name, total_time / cfg->NUM_RUNS);
    }
}

void is_symmetric_sequential_wrapper(float** matrix, int n, long double* time) {
    is_symmetric_sequential(matrix, n, time);
}

void is_symmetric_implicit_wrapper(float** matrix, int n, long double* time) {
    is_symmetric_implicit(matrix, n, time);
}

void is_symmetric_omp_wrapper(float** matrix, int n, long double* time) {
    is_symmetric_omp(matrix, n, time);
}

void transpose_sequential_wrapper(float** matrix, int n, long double* time) {
    float** result = transpose_sequential(matrix, n, time);
    deallocate_matrix(result, n);
}

void transpose_omp_wrapper(float** matrix, int n, long double* time) {
    float** result = transpose_omp(matrix, n, time);
    deallocate_matrix(result, n);
}

void transpose_omp_block_based_wrapper(float** matrix, int n, long double* time) {
    float** result = transpose_omp_block_based(matrix, n, get_config()->BLOCK_SIZE, time);
    deallocate_matrix(result, n);
}

void transpose_omp_tile_distributed_wrapper(float** matrix, int n, long double* time){
    float** result = transpose_omp_tile_distributed(matrix, n, get_config()->BLOCK_SIZE, time);
    deallocate_matrix(result, n);
}

void transpose_omp_tasks_wrapper(float** matrix, int n, long double* time) {
    float** result = transpose_omp_tasks(matrix, n, get_config()->BLOCK_SIZE, time);
    deallocate_matrix(result, n);
}
