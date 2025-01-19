#ifndef MPI_PARALLEL_H
#define MPI_PARALLEL_H

#include <mpi.h>
#include <stdbool.h>

bool is_symmetric_mpi(MPI_Comm comm, float** matrix, int n, int rank, int size, long double* time, int verbosity);

float** transpose_mpi(MPI_Comm comm, float** matrix, int n, int rank, int size, long double* time, int verbosity);

float** alltoall_transpose_mpi(MPI_Comm comm, float** matrix, int n, int rank, int size, long double* time, int verbosity);

#endif // !MPI_PARALLEL_H
