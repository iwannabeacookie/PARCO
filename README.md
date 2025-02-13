## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Building the Project](#building-the-project)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Benchmark](#running-the-benchmark)
  - [Generating Plots](#generating-plots)
- [Code Overview](#code-overview)

## Features

- **Matrix Operations**: Initialize, transpose, and check the symmetry of matrices.
- **Parallel Computing**: Utilize OpenMP and MPI for multi-threaded execution to enhance performance.
- **Benchmarking**: Measure and compare the performance of various matrix operation implementations.
- **Visualization**: Generate 2D plots to visualize benchmark results and speedups.
- **Configurability**: Easily adjust matrix dimensions, thread counts, block sizes, and verbosity levels via configuration.

## Project Structure

```
PARCO/
├── include/
│   ├── init_matrix.h
│   ├── sequential.h
│   ├── omp_parallel.h
│   ├── implicit_parallel.h
│   ├── config.h
│   ├── mpi_parallel.h
│   └── utils.h
├── src/
│   ├── sequential.c
│   ├── init_matrix.c
│   ├── omp_parallel.c
│   ├── implicit_parallel.c
│   ├── main.c
│   ├── mpi_parallel.c
│   ├── config.c
│   └── utils.c
├── README.md
├── create_venv.sh
├── start.pbs
├── Makefile
└── parser.py
```

### Directory Breakdown

- **include/**: Contains all the header files defining interfaces for various modules.
- **src/**: Holds the source `.c` files implementing the functionalities declared in the headers.
- **Makefile**: Automates the build process, handling compilation and linking.
- **create_venv.sh**: Shell script to create a virtual environment and install python dependencies.
- **parser.py**: Python script for parsing benchmark results and generating visualizations.
- **start.pbs**: PBS script for running the benchmark on the Unitn HPC cluster.

## Toolchain (Unitn HPC Cluster)
- **Compiler**: GCC 9.1.0
- **MPI**: MPICH 3.2.1
- **GNU Make**: 3.82
- **Python**: 3.10.14 + deps

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/iwannabeacookie/PARCO-D1
   cd PARCO-D1
   ```

## Running on the Unitn HPC Cluster

### Submitting the job

1. Make sure the repository is cloned into the home directory. Or change the pbs script to properly access the cloned repository.

2. **Queue the pbs script**

   ```bash
   qsub start.pbs
   ```

3. To add configuration to the job, add the "-v" flag and specify a string of [configuration](#configuration) options as a "flags" variable.

   ```bash
   qsub -v flags="CONFIGURATION" start.pbs
   ```

   **Example:**

   ```bash
   qsub -v flags="--runs 100 --matrix-dimension 64-2048 --threads 4 --block-size 32-256 --verbose 1" start.pbs
   ```

### Running Interactive

1. **Request an interactive session**

   ```bash
   qsub -I -l select=1:ncpus=4:mpiprocs=4:mem=1gb
   ```
2. **Proceed with the next subsection**

## Building the Project

The project uses a `Makefile` to manage the build process. The `Makefile` supports different build modes, including default, verbose, and debug.

### Default Build

To compile the project with default settings:

```bash
make all
```

This will generate the executable at `bin/out`.

### Verbose Build

For a build with additional compiler warnings and optimization reports:

```bash
make verbose
```

### Debug Build

To compile the project with debugging symbols:

```bash
make debug
```

### Cleaning Build Files

To remove generated binaries and object files:

```bash
make clean
```

## Usage

### Configuration

The project can be configured via command-line arguments that are parsed in `main.c` using the `init_config` function. Key configurable parameters include:

- **Matrix Dimensions**: Define the range from `MIN_MATRIX_DIMENSION` to `MAX_MATRIX_DIMENSION`, doubling each iteration.
- **Threads**: Set the range from `MIN_OMP_THREADS` to `MAX_OMP_THREADS`. Used both for OMP and MPI (cast as processes).
- **Block Size**: Specify the range for `BLOCK_SIZE`, doubling each iteration.
- **Verbosity Level**: Control the level of detail in the program's output.

### Running the Benchmark

**OpenMP** | After building the project, execute the benchmark as follows:

```bash
./bin/out [OPTIONS]
```

Use the '--help' flag to view the available options:

```bash
./bin/out --help
```

**MPI** | To run the MPI version, use the following command:

```bash
mpirun ./bin/out [OPTIONS]
```

**Example:**

```bash
./bin/out --runs 100 --matrix-dimension 64-2048 --threads 4 --block-size 32-256 --verbose 1
```

This command initializes the benchmark with matrix dimensions ranging from 64 to 2048, 4 threads, block sizes from 32 to 256, and sets the verbosity level to 1.

### Generating Plots

Once the benchmark completes, a CSV file named `benchmark_results.csv` is generated. Use the provided Python script to parse this data and generate visualizations.

1. **Create the Virtual Environment**

   ```bash
   ./create_venv.sh
   ```

1.1 The provided script will create a virtual environment named `venv`, install the required dependencies, and activate the environment. If you spawn another shell, you will have to activate the environment manually, before proceeding.

   ```bash
   source venv/bin/activate
   ```

2. **Run the Parser Script**

   Ensure you are in the project's root directory and execute:

   ```bash
   python parser.py
   ```

3. **Generated Plots**

   The script will generate various plots saved in the `plots/` directory, organized by matrix dimension and function names, as well as a `speedup_plot.png` summarizing the speedup metrics.

## Code Overview

### Key Components

- **Matrix Initialization (`init_matrix.c` & `init_matrix.h`)**
  - Functions to initialize matrices either sequentially or in parallel.

- **Matrix Operations**
  - **Sequential (`sequential.c` & `sequential.h`)**: Implements matrix operations without parallelization.
  - **OpenMP Parallel (`omp_parallel.c` & `omp_parallel.h`)**: Implements parallelized versions using OpenMP.
  - **Implicit Parallel (`implicit_parallel.c` & `implicit_parallel.h`)**: Utilizes compiler directives for implicit parallelization.
  - **MPI Parallel (`mpi_parallel.c` & `mpi_parallel.h`)**: Implements parallelized versions using MPI.
  
- **Configuration Management (`config.c` & `config.h`)**
  - Handles the parsing and management of configuration parameters.

- **Utility Functions (`utils.c` & `utils.h`)**
  - Includes helper functions for matrix printing, correctness checks, timing, and more.

- **Benchmarking (`main.c`)**
  - Orchestrates the benchmarking process by iterating over configurations, executing benchmarks, and recording results.

- **Makefile**
  - Automates the build process with support for different build modes.

- **Parser Script (`parser.py`)**
  - Parses the benchmark CSV results and generates 2D and 3D plots for performance visualization.

---

*Happy Computing!*
