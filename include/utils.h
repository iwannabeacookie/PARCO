#ifndef UTILS_H
#define UTILS_H

void print_matrix(float** matrix, int n);

void correct_transpose(float** m1, float** m2, int n);

void test_randomness(float** m1, float** m2, int n);

double get_time_in_seconds();

void benchmark_function(void (*func)(float**, int, long double*), float** matrix, int n, const char* func_name);

void is_symmetric_wrapper(float** matrix, int n, long double* time);

void is_symmetric_implicit_wrapper(float** matrix, int n, long double* time);

void is_symmetric_omp_wrapper(float** matrix, int n, long double* time);

void transpose_wrapper(float** matrix, int n, long double* time);

void transpose_omp_wrapper(float** matrix, int n, long double* time);

void transpose_omp_block_based_wrapper(float** matrix, int n, long double* time);

#endif // !UTILS_H
