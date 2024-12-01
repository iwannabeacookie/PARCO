#ifndef IMPLICIT_PARALLEL_H
#define IMPLICIT_PARALLEL_H

#include <stdbool.h>

bool is_symmetric_implicit(float **matrix, int n, long double* time);

float** transpose_implicit(float **matrix, int n, long double* time);

float** transpose_implicit_block_based(float **matrix, int n, long double* time);

float** transpose_implicit_cache_oblivious(float **matrix, int n, long double* time);

#endif // !IMPLICIT_PARALLEL_H
