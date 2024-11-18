#include <omp.h>
#include <stdio.h>
#include "../include/init_matrix.h"
#include "../include/sequential.h"

#define MATRIX_DIMENSION 10000

void print_matrix(float** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    float** m = init_matrix_sequential(MATRIX_DIMENSION);

    // print_matrix(m, MATRIX_DIMENSION);

    is_symmetric(m, MATRIX_DIMENSION);

    float ** t = transpose(m, MATRIX_DIMENSION);

    float ** i_t = transpose_implicit(m, MATRIX_DIMENSION);

    // print_matrix(t, MATRIX_DIMENSION);

    return 0;
}
