#include <omp.h>
#include <stdio.h>
#include <time.h>
#include "../include/init_matrix.h"
#include "../include/sequential.h"
#include "../include/parallel.h"
#include "../include/utils.h"

#define MATRIX_DIMENSION 4096
#define NUM_RUNS 10

int main(int argc, char *argv[]) {
    printf("===== Matrix Generation =====\n");
    float** p = init_matrix_sequential(MATRIX_DIMENSION);
    float** m = init_matrix_parallel(MATRIX_DIMENSION);

    printf("\n===== Symmetricity Checks =====\n");
    benchmark_function(is_symmetric_wrapper, m, MATRIX_DIMENSION, "is_symmetric");
    benchmark_function(is_symmetric_implicit_wrapper, m, MATRIX_DIMENSION, "is_symmetric_implicit");
    benchmark_function(is_symmetric_omp_wrapper, m, MATRIX_DIMENSION, "is_symmetric_omp");

    printf("\n===== Transpositions =====\n");
    benchmark_function(transpose_wrapper, m, MATRIX_DIMENSION, "transpose");
    benchmark_function(transpose_omp_wrapper, m, MATRIX_DIMENSION, "transpose_omp");
    benchmark_function(transpose_omp_block_based_wrapper, m, MATRIX_DIMENSION, "transpose_omp_block_based");

    return 0;
}
