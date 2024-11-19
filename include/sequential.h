#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <stdbool.h>

bool is_symmetric(float **matrix, int n);

bool is_symmetric_implicit(float **matrix, int n);

float** transpose(float **matrix, int n);

float** transpose_implicit(float **matrix, int n);

#endif // !SEQUENTIAL_H
