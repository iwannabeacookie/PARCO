#ifndef PARALLEL_H
#define PARALLEL_H

#include <stdbool.h>

bool is_symmetric_omp(float **matrix, int n, long double* time);

float** transpose_omp(float **matrix, int n, long double* time);

float** transpose_omp_block_based(float **matrix, int n, int block_size, long double* time);

float** transpose_omp_tile_distributed(float **matrix, int n, int tile_size, long double* time);

float ** transpose_omp_tasks(float **matrix, int n, int tile_size, long double* time);

#endif // !PARALLEL_H
