#include <omp.h>
#include <stdio.h>
#include "../include/init_matrix.h"
#include "../include/sequential.h"
#include "../include/parallel.h"

#define MATRIX_DIMENSION 4096

void print_matrix(float** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void correct_transpose(float** m1, float** m2, int n) {
    int equal = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            equal = m1[i][j] == m2[i][j];
            if (!equal) {
                printf("Not correct\n");
                return;
            }
        }
    }
    printf("Correct\n");
}

int main(int argc, char *argv[])
{
    float** m = init_matrix_parallel(MATRIX_DIMENSION);

    // print_matrix(m, MATRIX_DIMENSION);

    is_symmetric(m, MATRIX_DIMENSION);

    is_symmetric_implicit(m, MATRIX_DIMENSION);

    is_symmetric_omp(m, MATRIX_DIMENSION);

    float ** t = transpose(m, MATRIX_DIMENSION);
    // print_matrix(t, MATRIX_DIMENSION);

    float ** i_t = transpose_implicit(m, MATRIX_DIMENSION);
    // print_matrix(i_t, MATRIX_DIMENSION);
    // correct_transpose(t, i_t, MATRIX_DIMENSION);

    float ** o_t = transpose_omp(m, MATRIX_DIMENSION);
    // print_matrix(o_t, MATRIX_DIMENSION);
    // correct_transpose(t, o_t, MATRIX_DIMENSION);

    float ** b_t = transpose_omp_block_based(m, MATRIX_DIMENSION);
    // print_matrix(b_t, MATRIX_DIMENSION);
    // correct_transpose(t, b_t, MATRIX_DIMENSION);

    return 0;
}
