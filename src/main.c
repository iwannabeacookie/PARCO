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

    for (int size = cfg->MIN_MATRIX_DIMENSION; size <= cfg->MAX_MATRIX_DIMENSION; size *= 2) {
        for (int threads = cfg->MIN_OMP_THREADS; threads <= cfg->MAX_OMP_THREADS; threads++) {
            cfg->MATRIX_DIMENSION = size;
            cfg->OMP_THREADS = threads;
            omp_set_num_threads(threads);

            printf("===== Processing Matrix Generation for size %d with %d threads =====\n", size, threads);
            float** p = init_matrix_sequential(cfg->MATRIX_DIMENSION);
            float** m = init_matrix_parallel(cfg->MATRIX_DIMENSION);

            cfg->MATRIX = p;

            deallocate_matrix(m, cfg->MATRIX_DIMENSION);

            printf("\n===== Processing Symmetricity Checks for size %d with %d threads =====\n", size, threads);
            benchmark_function(is_symmetric_sequential_wrapper, "is_symmetric_sequential");
            benchmark_function(is_symmetric_implicit_wrapper, "is_symmetric_implicit");
            benchmark_function(is_symmetric_omp_wrapper, "is_symmetric_omp");

            printf("\n===== Processing Transpositions for size %d with %d threads =====\n", size, threads);
            benchmark_function(transpose_sequential_wrapper, "transpose_sequential");
            benchmark_function(transpose_implicit_wrapper, "transpose_implicit");

            for (int block_size = cfg->MIN_BLOCK_SIZE; block_size <= cfg->MAX_BLOCK_SIZE; block_size *= 2) {
                cfg->BLOCK_SIZE = block_size;
                printf("\n--- Block Size: %d ---\n", block_size);
                benchmark_function(transpose_implicit_block_based_wrapper, "transpose_implicit_block_based");
                benchmark_function(transpose_impplicit_cache_oblivious_wrapper, "transpose_implicit_cache_oblivious");
                benchmark_function(transpose_omp_wrapper, "transpose_omp");
                benchmark_function(transpose_omp_block_based_wrapper, "transpose_omp_block_based");
                benchmark_function(transpose_omp_tile_distributed_wrapper, "transpose_omp_tile_distributed");
                benchmark_function(transpose_omp_tasks_wrapper, "transpose_omp_tasks");

                if (cfg->VERBOSE_LEVEL > 0) {
                    printf("\n %%- Cache-Oblivious Transposition -%%\n");
                }
                benchmark_function(transpose_cache_oblivious_wrapper, "transpose_cache_oblivious");
            }
            

            printf("\n");
        }
        deallocate_matrix(cfg->MATRIX, cfg->MATRIX_DIMENSION);
    }

    return 0;
}

