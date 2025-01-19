#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include "../include/init_matrix.h"
#include "../include/utils.h"
#include "../include/config.h"


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    init_config(argc, argv);
    Config* cfg = get_config();
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    for (int size = cfg->MIN_MATRIX_DIMENSION; size <= cfg->MAX_MATRIX_DIMENSION; size *= 2) {
        for (int mpi_procs = cfg->MIN_OMP_THREADS; mpi_procs <= cfg->MAX_OMP_THREADS && mpi_procs <= world_size; mpi_procs *= 2) {
            MPI_Comm sub_comm;
            int color = world_rank < mpi_procs ? 0 : MPI_UNDEFINED;
            MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);
            cfg->CURR_COMM = sub_comm;
            

            if (color == 0) {
                int threads = mpi_procs;
                cfg->MATRIX_DIMENSION = size;
                cfg->OMP_THREADS = threads;
                omp_set_num_threads(threads);

                if (world_rank == 0) {
                    printf("===== Processing Matrix Generation for size %d with %d threads and %d MPI processes =====\n", size, threads, mpi_procs);
                    float** p = init_matrix_sequential(cfg->MATRIX_DIMENSION);
                    float** m = init_matrix_parallel(cfg->MATRIX_DIMENSION);

                    cfg->MATRIX = p;

                    deallocate_matrix(m, cfg->MATRIX_DIMENSION);

                    printf("\n===== Processing Symmetricity Checks for size %d with %d threads and %d MPI processes =====\n", size, threads, mpi_procs);
                    benchmark_function(is_symmetric_sequential_wrapper, "is_symmetric_sequential");
                    benchmark_function(is_symmetric_implicit_wrapper, "is_symmetric_implicit");
                    benchmark_function(is_symmetric_omp_wrapper, "is_symmetric_omp");

                    printf("\n===== Processing Transpositions for size %d with %d threads and %d MPI processes =====\n", size, threads, mpi_procs);
                    benchmark_function(transpose_sequential_wrapper, "transpose_sequential");
                    benchmark_function(transpose_implicit_wrapper, "transpose_implicit");

                    for (int block_size = cfg->MIN_BLOCK_SIZE; block_size <= cfg->MAX_BLOCK_SIZE; block_size *= 2) {
                        cfg->BLOCK_SIZE = block_size;
                        printf("\n--- Block Size: %d ---\n", block_size);
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

                MPI_Barrier(cfg->CURR_COMM);

                if (cfg->VERBOSE_LEVEL > 0 && world_rank == 0) {
                    printf("\n %%- MPI Symmetricity Check -%%\n");
                }
                benchmark_function(is_symmetric_mpi_wrapper, "is_symmetric_mpi");

                if (cfg->VERBOSE_LEVEL > 0 && world_rank == 0) {
                    printf("\n %%- MPI Transposition -%%\n");
                }
                benchmark_function(transpose_mpi_wrapper, "transpose_mpi");

                if (cfg->VERBOSE_LEVEL > 0 && world_rank == 0) {
                    printf("\n %%- MPI All to All Transposition -%%\n");
                }
                benchmark_function(alltoall_transpose_mpi_wrapper, "transpose_mpi");

                // if (cfg->VERBOSE_LEVEL > 0 && world_rank == 0) {
                //     printf("\n %%- MPI Transposition with 2D Decomposition -%%\n");
                // }

                MPI_Comm_free(&sub_comm);
                if (world_rank == 0) {
                    cfg->CURR_COMM = MPI_COMM_WORLD;
                    printf("\n");
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
