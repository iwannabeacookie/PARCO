#include "../include/utils.h"
#include <stdio.h>

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

// Don't use that function on bigger inputs
void test_randomness(float** m1, float** m2, int n) {
    int occurrences = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                for (int t = 0; t < n; t++) {
                    if (m1[i][j] == m2[k][t]) {
                        occurrences++;
                    }
                    if (occurrences > 1) {
                        printf("Not random\n");
                        return;
                    }
                }
            }
            occurrences = 0;
        }
    }
    printf("Random\n");
}
