#ifndef UTILS_H
#define UTILS_H

void print_matrix(float** matrix, int n);

void correct_transpose(float** m1, float** m2, int n);

void test_randomness(float** m1, float** m2, int n);

double get_time_in_seconds();

void print_loading_bar(int progress, int total);

void deallocate_matrix(float** matrix, int n);

void benchmark_function(void (*func)(long double*), const char* func_name);

void is_symmetric_sequential_wrapper(long double* time);

void is_symmetric_implicit_wrapper(long double* time);

void is_symmetric_omp_wrapper(long double* time);

void transpose_sequential_wrapper(long double* time);

void transpose_implicit_wrapper(long double* time);

void transpose_omp_wrapper(long double* time);

void transpose_omp_block_based_wrapper(long double* time);

void transpose_omp_tile_distributed_wrapper(long double* time);

void transpose_omp_tasks_wrapper(long double* time);

void transpose_cache_oblivious_wrapper(long double* time);

#endif // !UTILS_H
