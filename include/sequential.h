#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <stdbool.h>

bool is_symmetric_sequential(float **matrix, int n, long double* time);

float** transpose_sequential(float **matrix, int n, long double* time);

#endif // !SEQUENTIAL_H
