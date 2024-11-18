#include <stdbool.h>

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

bool is_symmetric(float **matrix, int n);

float** transpose(float **matrix, int n);

float** transpose_implicit(float **matrix, int n);

#endif // !SEQUENTIAL_H
