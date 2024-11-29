#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "../include/init_matrix.h"
#include "../include/sequential.h"
#include "../include/omp_parallel.h"
#include "../include/utils.h"
#include "../include/config.h"

int main(int argc, char *argv[]) {
    init_config(argc, argv);

    Config* cfg = get_config();

    printf("===== Matrix Generation =====\n");
    float** p = init_matrix_sequential(cfg->MATRIX_DIMENSION);
    float** m = init_matrix_parallel(cfg->MATRIX_DIMENSION);

    printf("\n===== Symmetricity Checks =====\n");
    benchmark_function(is_symmetric_sequential_wrapper, m, cfg->MATRIX_DIMENSION, "is_symmetric_sequential");
    benchmark_function(is_symmetric_implicit_wrapper, m, cfg->MATRIX_DIMENSION, "is_symmetric_implicit");
    benchmark_function(is_symmetric_omp_wrapper, m, cfg->MATRIX_DIMENSION, "is_symmetric_omp");

    printf("\n===== Transpositions =====\n");
    benchmark_function(transpose_sequential_wrapper, m, cfg->MATRIX_DIMENSION, "transpose_sequential");
    benchmark_function(transpose_omp_wrapper, m, cfg->MATRIX_DIMENSION, "transpose_omp");
    benchmark_function(transpose_omp_block_based_wrapper, m, cfg->MATRIX_DIMENSION, "transpose_omp_block_based");

    return 0;
}
