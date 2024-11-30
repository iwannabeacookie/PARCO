#include "../include/config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>

// Static instance of Config
Config config;

// Initializes the configuration, possibly from a file
void init_config(int argc, char *argv[]) {
    // Default values
    int CURR_RUN = 0;
    int MATRIX_DIMENSION = 1024;
    int VERBOSE_LEVEL = 0;
    int BLOCK_SIZE = 4;
    int NUM_RUNS = 1; 
    int BENCHMARK_SEQUENTIAL = 0; 
    int BENCHMARK_IMPLICIT = 0;
    int BENCHMARK_OMP = 0; 
    int OMP_THREADS = 4; 

    printf("argc: %d\n", argc);
    // Print argv
    for (int i = 0; i < argc; i++){
        printf("argv[%d]: %s\n", i, argv[i]);
    }


    // Set the execution parameters
    for (int i = 1; i < argc; i++){
        // Print help
        if (strcmp(argv[i], "--help") == 0){
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --help                     Print this message\n");
            printf("  --verbose <int>            Set the verbose level (default: 0)\n");
            printf("  --matrix-dimension <int>  Set the matrix dimension (default: 1024)\n");
            printf("  --block-size <int>         Set the block size (default: 4)\n");
            printf("  --runs <int>               Set the number of runs (default: 1)\n");
            printf("  --threads <int>            Set the number of threads (default: 4)\n");
            printf("  -s                         Run the sequential benchmark\n");
            printf("  -i                         Run the implicit benchmark\n");
            printf("  -o                         Run the OpenMP benchmark\n");
            exit(0);
        }

        // Set the matrix dimension
        else if (strcmp(argv[i], "--matrix-dimension") == 0){
            if (i + 1 < argc){
                MATRIX_DIMENSION = atoi(argv[i + 1]);
                i++;
            }
            else{
                fprintf(stderr, "%s", "Error: --matrix-dimension flag requires an argument\n");
                exit(1);
            }
        }

        // Set the verbose level
        else if (strcmp(argv[i], "--verbose") == 0){
            if (i + 1 < argc){
                int verbose_level = atoi(argv[i + 1]);
                if (verbose_level < 0 || verbose_level > 2){
                    fprintf(stderr, "%s", "Error: Verbose level must be between 0 and 2\n");
                    exit(1);
                }
                VERBOSE_LEVEL = verbose_level;
                i++;
            }
            else{
                fprintf(stderr, "%s", "Error: --verbose flag requires an argument\n");
                exit(1);
            }
        }

        // Set the block size
        else if (strcmp(argv[i], "--block-size") == 0){
            if (i + 1 < argc){
                BLOCK_SIZE = atoi(argv[i + 1]);
                i++;
            }
            else{
                fprintf(stderr, "%s", "Error: --block-size flag requires an argument\n");
                exit(1);
            }
        }

        // Set the number of runs
        else if (strcmp(argv[i], "--runs") == 0){
            if (i + 1 < argc){
                NUM_RUNS = atoi(argv[i + 1]);
                if (NUM_RUNS <= 0){
                    fprintf(stderr, "%s", "Error: Number of runs must be greater than 0\n");
                    exit(1);
                }
                i++;
            }
            else{
                fprintf(stderr, "%s", "Error: --runs flag requires an argument\n");
                exit(1);
            }
        }

        // Select benchmarks to run
        else if ((argv[i][0] == '-') && (argv[i][1] != '-')) {
            for (int j = 1; j < (int)strlen(argv[i]); j++) {
                switch (argv[i][j]) {
                    case 's':
                        BENCHMARK_SEQUENTIAL = 1;
                        break;
                    case 'i':
                        BENCHMARK_IMPLICIT = 1;
                        break;
                    case 'o':
                        BENCHMARK_OMP = 1;
                        break;
                    default:
                        fprintf(stderr, "Error: Invalid test option '%c'\n", argv[i][j]);
                        exit(1);
                }
            }
        }

        else if (strcmp(argv[i], "--threads") == 0){
            if (i + 1 < argc){
                OMP_THREADS = atoi(argv[i + 1]);
                if (OMP_THREADS <= 0){
                    fprintf(stderr, "%s", "Error: Number of threads must be greater than 0\n");
                    exit(1);
                }
                i++;
            }
            else{
                fprintf(stderr, "%s", "Error: --threads flag requires an argument\n");
                exit(1);
            }

        }
        else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            exit(1);
        }
    }

    // Set the configuration
    config.CURR_RUN = CURR_RUN;
    config.MATRIX_DIMENSION = MATRIX_DIMENSION;
    config.VERBOSE_LEVEL = VERBOSE_LEVEL;
    config.BLOCK_SIZE = BLOCK_SIZE;
    config.NUM_RUNS = NUM_RUNS;
    config.BENCHMARK_SEQUENTIAL = BENCHMARK_SEQUENTIAL;
    config.BENCHMARK_IMPLICIT = BENCHMARK_IMPLICIT;
    config.BENCHMARK_OMP = BENCHMARK_OMP;
    config.OMP_THREADS = OMP_THREADS;
}

Config* get_config() {
    return &config;
}
