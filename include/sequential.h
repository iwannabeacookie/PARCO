#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <stdbool.h>

bool is_symmetric(float **matrix, int n, long double* time);

bool is_symmetric_implicit(float **matrix, int n, long double* time);

float** transpose(float **matrix, int n, long double* time);

float** transpose_implicit(float **matrix, int n, long double* time);

#endif // !SEQUENTIAL_H
