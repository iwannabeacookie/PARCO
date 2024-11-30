#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    int CURR_RUN;
    float** MATRIX;
    int MIN_MATRIX_DIMENSION;
    int MAX_MATRIX_DIMENSION;
    int MATRIX_DIMENSION;
    int VERBOSE_LEVEL;
    int MIN_BLOCK_SIZE;
    int MAX_BLOCK_SIZE;
    int BLOCK_SIZE;
    int NUM_RUNS;
    int BENCHMARK_FULL;
    int MIN_OMP_THREADS;
    int MAX_OMP_THREADS;
    int OMP_THREADS;
} Config;

// Function to initialize the configuration
void init_config(int argc, char *argv[]);

// Function to retrieve the configuration
Config* get_config();

#endif // !CONFIG_H
