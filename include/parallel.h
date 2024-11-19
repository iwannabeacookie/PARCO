#ifndef PARALLEL_H
#define PARALLEL_H

#include <stdbool.h>

bool is_symmetric_omp(float **matrix, int n);

float** transpose_omp(float **matrix, int n);

float** transpose_omp_block_based(float **matrix, int n);

#endif // !PARALLEL_H
