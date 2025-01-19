#include "../include/config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

// Static instance of Config
Config config;

bool is_power_of_two(int n) {
    return (n & (n - 1)) == 0;
}

void parse_range(const char *arg, int *mn, int *mx) {
    char *dash = strchr(arg, '-');
    if (dash) {
        int min = atoi(arg);
        int max = atoi(dash + 1);

        if (min > max) {
            fprintf(stderr, "Error: Invalid range %s\n", arg);
            exit(1);
        }

        if (!is_power_of_two(min) || !is_power_of_two(max)) {
            fprintf(stderr, "Error: Range values must be power of two\n");
            exit(1);
        }

        *mn = min;
        *mx = max;
    } else {
        *mn = *mx = atoi(arg);
    }
}

// Initializes the configuration, possibly from a file
void init_config(int argc, char *argv[]) {
    // Default values
    int CURR_RUN = 0;
    int MATRIX_DIMENSION = 1024;
    int VERBOSE_LEVEL = 0;
    int BLOCK_SIZE = 4;
    int NUM_RUNS = 1;
    int BENCHMARK_FULL = 0;
    int OMP_THREADS = 4;

    int MIN_MATRIX_DIMENSION = MATRIX_DIMENSION;
    int MAX_MATRIX_DIMENSION = MATRIX_DIMENSION;
    int MIN_BLOCK_SIZE = BLOCK_SIZE;
    int MAX_BLOCK_SIZE = BLOCK_SIZE;
    int MIN_OMP_THREADS = OMP_THREADS;
    int MAX_OMP_THREADS = OMP_THREADS;

    // Set the execution parameters
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --help                     Print this message\n");
            printf("  --verbose <int>            Set the verbose level (default: 0)\n");
            printf("  --runs <int>               Set the number of runs (default: 1)\n");
            printf("  --benchmark-full           Benchmark full ranges (ranges options will be disabled)\n");
            printf("\n");
            printf("Range options (must be expressed in powers of two):\n");
            printf("  --matrix-dimension <int|int-int>   Set the matrix dimension (default: 1024) (full: 2^4 - 2^12)\n");
            printf("  --block-size <int|int-int>         Set the block size (default: 4) (full: 2^2 - 2^8)\n");
            printf("  --threads <int|int-int>            Set the number of threads (default: 4) (full: 2^1 - 2^3)\n");
            printf("\n");
            printf("WARNING: Running with matrix-dimension larger than the full range is not allowed due to project specifications\n");
            printf("         Running with block-size larger than the full range does not make much sense as the value should be optimized to the cache size\n");
            printf("         Running with threads larger than the full range is only recommended when running on a computing cluster\n");
            exit(0);
        } else if (strcmp(argv[i], "--matrix-dimension") == 0) {
            if (i + 1 < argc) {
                parse_range(argv[i + 1], &MIN_MATRIX_DIMENSION, &MAX_MATRIX_DIMENSION);
                MATRIX_DIMENSION = MIN_MATRIX_DIMENSION;

                if (MATRIX_DIMENSION < 16 || MAX_MATRIX_DIMENSION > 4096) {
                    fprintf(stderr, "Error: Matrix dimension must be between 16 and 4096\n");
                    exit(1);
                }

                i++;
            } else {
                fprintf(stderr, "Error: --matrix-dimension flag requires an argument\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--verbose") == 0) {
            if (i + 1 < argc) {
                int verbose_level = atoi(argv[i + 1]);
                if (verbose_level < 0 || verbose_level > 2) {
                    fprintf(stderr, "Error: Verbose level must be between 0 and 2\n");
                    exit(1);
                }
                VERBOSE_LEVEL = verbose_level;
                i++;
            } else {
                fprintf(stderr, "Error: --verbose flag requires an argument\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--block-size") == 0) {
            if (i + 1 < argc) {
                parse_range(argv[i + 1], &MIN_BLOCK_SIZE, &MAX_BLOCK_SIZE);
                BLOCK_SIZE = MIN_BLOCK_SIZE;

                if (BLOCK_SIZE <= 0) {
                    fprintf(stderr, "Error: Block size must be larger than 0\n");
                    exit(1);
                }

                i++;
            } else {
                fprintf(stderr, "Error: --block-size flag requires an argument\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--runs") == 0) {
            if (i + 1 < argc) {
                NUM_RUNS = atoi(argv[i + 1]);
                if (NUM_RUNS <= 0) {
                    fprintf(stderr, "Error: Number of runs must be greater than 0\n");
                    exit(1);
                }
                i++;
            } else {
                fprintf(stderr, "Error: --runs flag requires an argument\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--benchmark-full") == 0) {
            BENCHMARK_FULL = 1;
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                parse_range(argv[i + 1], &MIN_OMP_THREADS, &MAX_OMP_THREADS);
                OMP_THREADS = MIN_OMP_THREADS;

                if (OMP_THREADS < 2) {
                    fprintf(stderr, "Warning: Number of threads is less than 2: the program will not run concurrently\n");
                }

                i++;
            } else {
                fprintf(stderr, "Error: --threads flag requires an argument\n");
                exit(1);
            }
        } else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            exit(1);
        }
    }

    FILE *fp = fopen("benchmark_results.csv", "w");
    if (fp != NULL) {
        fprintf(fp, "matrix_dimension,threads,block_size,time,func_name\n");
        fclose(fp);
    }

    // Set the configuration
    config.CURR_RUN = CURR_RUN;
    config.MIN_MATRIX_DIMENSION = MIN_MATRIX_DIMENSION;
    config.MAX_MATRIX_DIMENSION = MAX_MATRIX_DIMENSION;
    config.MATRIX_DIMENSION = MATRIX_DIMENSION;
    config.VERBOSE_LEVEL = VERBOSE_LEVEL;
    config.MIN_BLOCK_SIZE = MIN_BLOCK_SIZE;
    config.MAX_BLOCK_SIZE = MAX_BLOCK_SIZE;
    config.BLOCK_SIZE = BLOCK_SIZE;
    config.NUM_RUNS = NUM_RUNS;
    config.BENCHMARK_FULL = BENCHMARK_FULL;
    config.MIN_OMP_THREADS = MIN_OMP_THREADS;
    config.MAX_OMP_THREADS = MAX_OMP_THREADS;
    config.OMP_THREADS = OMP_THREADS;

    if (config.BENCHMARK_FULL == 1) {
        printf("### Benchmarking full ranges ###\n");
        
        config.MIN_MATRIX_DIMENSION = 16;
        config.MAX_MATRIX_DIMENSION = 4096;

        config.MIN_BLOCK_SIZE = 4;
        config.MAX_BLOCK_SIZE = 256;

        config.MIN_OMP_THREADS = 1;
        config.MAX_OMP_THREADS = 64;
    }
}

Config* get_config() {
    return &config;
}
