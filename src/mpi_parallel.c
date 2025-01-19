#include "../include/mpi_parallel.h"
#include "../include/utils.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

// Function to flatten the 2D matrix into a 1D array
float* flatten_matrix(float** matrix, int n) {
    float* flat = malloc(n * n * sizeof(float));
    if (flat == NULL) {
        fprintf(stderr, "Failed to allocate memory for flat_matrix.\n");
        return NULL;
    }
    for(int i = 0; i < n; i++) {
        memcpy(&flat[i * n], matrix[i], n * sizeof(float));
    }
    return flat;
}

// Function to create a 2D matrix from a 1D flat array
float** create_2d_matrix(float* flat, int rows, int cols) {
    float** matrix = malloc(rows * sizeof(float*));
    if (matrix == NULL) {
        fprintf(stderr, "Failed to allocate memory for matrix pointers.\n");
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(float));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for matrix row %d.\n", i);
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
        memcpy(matrix[i], &flat[i * cols], cols * sizeof(float));
    }

    return matrix;
}

bool is_symmetric_mpi(MPI_Comm comm, float **matrix, int n, int rank, int size, long double *time, int verbosity) {
    float* flat_matrix = NULL;
    float* flat_transposed = NULL;

    // Enhanced Debugging Print
    if (verbosity >= 2) {
        printf("Starting is_symmetric_mpi on rank %d\n", rank);
        fflush(stdout);
    }

    // Ensure that n is divisible by size
    if (n % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size n=%d is not divisible by number of processes size=%d.\n", n, size);
        }
        return false;
    }

    // Calculate the number of rows per process
    int rows_per_proc = n / size; // Correctly based on size

    // Only the root process flattens the matrix
    if (rank == 0) {
        flat_matrix = flatten_matrix(matrix, n);
        if (flat_matrix == NULL) {
            fprintf(stderr, "Failed to flatten the matrix.\n");
            return false;
        }
    }

    // Allocate memory for the local chunk
    float* local_matrix = malloc(rows_per_proc * n * sizeof(float));
    if (local_matrix == NULL) {
        fprintf(stderr, "Failed to allocate memory for local_matrix on rank %d.\n", rank);
        if (rank == 0) free(flat_matrix);
        return false;
    }

    // Scatter the matrix rows to all processes
    int scatter_err = MPI_Scatter(
        flat_matrix,                    // send buffer (root)
        rows_per_proc * n,              // send count per process
        MPI_FLOAT,                      // send type
        local_matrix,                   // receive buffer
        rows_per_proc * n,              // receive count
        MPI_FLOAT,                      // receive type
        0,                              // root
        comm                            // communicator
    );

    if (scatter_err != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Scatter failed on rank %d.\n", rank);
        if (rank == 0) free(flat_matrix);
        free(local_matrix);
        return false;
    }

    // Start timing after scattering
    MPI_Barrier(comm); // Ensure all processes have received their data
    double start_time = MPI_Wtime();

    // Transpose the local chunk
    float* local_transposed = malloc(rows_per_proc * n * sizeof(float));
    if (local_transposed == NULL) {
        fprintf(stderr, "Failed to allocate memory for local_transposed on rank %d.\n", rank);
        if (rank == 0) free(flat_matrix);
        free(local_matrix);
        return false;
    }

    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            local_transposed[j * rows_per_proc + i] = local_matrix[i * n + j];
        }
    }

    // Gather the transposed chunks back to the root
    if (rank == 0) {
        flat_transposed = malloc(n * n * sizeof(float));
        if (flat_transposed == NULL) {
            fprintf(stderr, "Failed to allocate memory for flat_transposed on root.\n");
            free(flat_matrix);
            free(local_matrix);
            free(local_transposed);
            return false;
        }
    }

    int gather_err = MPI_Gather(
        local_transposed,               // send buffer
        rows_per_proc * n,              // send count
        MPI_FLOAT,                      // send type
        flat_transposed,                // receive buffer (root)
        rows_per_proc * n,              // receive count
        MPI_FLOAT,                      // receive type
        0,                              // root
        comm                            // communicator
    );

    if (gather_err != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Gather failed on rank %d.\n", rank);
        if (rank == 0) free(flat_matrix);
        if (rank == 0) free(flat_transposed);
        free(local_matrix);
        free(local_transposed);
        return false;
    }

    // End timing after gathering
    double end_time = MPI_Wtime();
    *time = end_time - start_time;

    // Check symmetry on the root process
    bool is_symmetric = true;
    if (rank == 0) {
        for (int i = 0; i < n && is_symmetric; i++) {
            for (int j = 0; j < n && is_symmetric; j++) {
                if (flat_matrix[i * n + j] != flat_transposed[i * n + j]) {
                    is_symmetric = false;
                }
            }
        }
        free(flat_transposed);
    }

    // Broadcast the result to all processes
    MPI_Bcast(&is_symmetric, 1, MPI_C_BOOL, 0, comm);

    // Cleanup
    if (rank == 0) {
        free(flat_matrix);
    }
    free(local_matrix);
    free(local_transposed);

    // Enhanced Debugging Print
    if (verbosity >= 2) {
        printf("is_symmetric_mpi completed successfully on rank %d\n", rank);
        fflush(stdout);
    }

    return is_symmetric;
}

float** transpose_mpi(MPI_Comm comm, float** matrix, int n, int rank, int size, long double* time, int verbosity) {
    float* flat_matrix = NULL;
    float* flat_transposed = NULL;

    // Enhanced Debugging Print
    if (verbosity >= 2) {
        printf("Rank %d: Starting transpose_mpi\n", rank);
        fflush(stdout);
    }

    // Only root process flattens the matrix
    if(rank == 0) {
        flat_matrix = flatten_matrix(matrix, n);
        if (flat_matrix == NULL) {
            fprintf(stderr, "Rank %d: Failed to flatten the matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    } else {
        // Allocate a dummy buffer on non-root processes to avoid passing NULL
        flat_matrix = malloc(1); // Small buffer since it's not used
        if (flat_matrix == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate dummy buffer for flat_matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Calculate the number of rows per process
    int rows_per_proc = n / size; // Guaranteed to be integer since n and size are powers of two and size < n
    // No remainder due to n divisible by size and both being powers of two

    // Allocate memory for the local chunk
    float* local_matrix = malloc(rows_per_proc * n * sizeof(float));
    if (local_matrix == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate memory for local_matrix.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Debugging Print before Scatter
    if (verbosity >= 2) {
        printf("Rank %d: Scattering data\n", rank);
        fflush(stdout);
    }

    // Scatter the matrix rows to all processes
    int scatter_err = MPI_Scatter(
        flat_matrix,                    // send buffer (root)
        rows_per_proc * n,              // send count per process
        MPI_FLOAT,                      // send type
        local_matrix,                   // receive buffer
        rows_per_proc * n,              // receive count
        MPI_FLOAT,                      // receive type
        0,                              // root
        comm                  // communicator
    );

    if (scatter_err != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: MPI_Scatter failed.\n", rank);
        MPI_Abort(comm, scatter_err);
    }

    // Free the dummy buffer on non-root processes
    if(rank != 0){
        free(flat_matrix);
    }

    // Debugging Print after Scatter
    if (verbosity >= 2) {
        printf("Rank %d: Data scattered\n", rank);
        fflush(stdout);
    }

    // Measure the start time
    double start_time = MPI_Wtime();

    // Transpose the local chunk
    float* local_transposed = malloc(rows_per_proc * n * sizeof(float));
    if (local_transposed == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate memory for local_transposed.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    for(int i = 0; i < rows_per_proc; i++) {
        for(int j = 0; j < n; j++) {
            // Correct indexing for transposed data
            local_transposed[j * rows_per_proc + i] = local_matrix[i * n + j];
        }
    }

    // Measure the end time
    double end_time = MPI_Wtime();
    *time = end_time - start_time;

    // Debugging Print after Local Transpose
    if (verbosity >= 2) {
        printf("Rank %d: Local transposition completed in %Lf seconds\n", rank, *time);
        fflush(stdout);
    }

    // Allocate memory for the transposed flat matrix on root
    if(rank == 0) {
        flat_transposed = malloc(n * n * sizeof(float));
        if (flat_transposed == NULL) {
            fprintf(stderr, "Root Rank %d: Failed to allocate memory for flat_transposed.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Debugging Print before Gather
    if (verbosity >= 2) {
        printf("Rank %d: Gathering transposed data\n", rank);
        fflush(stdout);
    }

    // Gather the transposed chunks back to the root
    int gather_err = MPI_Gather(
        local_transposed,               // send buffer
        rows_per_proc * n,              // send count
        MPI_FLOAT,                      // send type
        flat_transposed,                // receive buffer (root)
        rows_per_proc * n,              // receive count
        MPI_FLOAT,                      // receive type
        0,                              // root
        comm                  // communicator
    );

    if (gather_err != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: MPI_Gather failed.\n", rank);
        MPI_Abort(comm, gather_err);
    }

    // Debugging Print after Gather
    if (verbosity >= 2) {
        printf("Rank %d: Transposed data gathered\n", rank);
        fflush(stdout);
    }

    // Root process reconstructs the transposed matrix
    float** transposed = NULL;
    if(rank == 0) {
        transposed = create_2d_matrix(flat_transposed, n, n);
        if (transposed == NULL) {
            fprintf(stderr, "Root Rank %d: Failed to create transposed matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
        // Free flattened arrays
        free(flat_matrix);
        free(flat_transposed);
    }

    // Cleanup
    free(local_matrix);
    free(local_transposed);

    // Debugging Print at End
    if (verbosity >= 2) {
        printf("Rank %d: transpose_mpi completed successfully\n", rank);
        fflush(stdout);
    }

    return transposed;
}

float** alltoall_transpose_mpi(MPI_Comm comm, float** matrix, int n, int rank, int size, long double* time, int verbosity) {
    float* flat_matrix = NULL;
    float* transposed_flat = NULL;

    // Enhanced Debugging Print
    if (verbosity >= 2) {
        printf("Rank %d: Starting alltoall_transpose_mpi\n", rank);
        fflush(stdout);
    }

    // Only root process flattens and broadcasts the matrix
    if(rank == 0) {
        flat_matrix = flatten_matrix(matrix, n);
        if (flat_matrix == NULL) {
            fprintf(stderr, "Rank %d: Failed to flatten the matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    } else {
        flat_matrix = malloc(n * n * sizeof(float));
        if (flat_matrix == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate buffer for flat_matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Broadcast the flattened matrix to all processes
    MPI_Bcast(flat_matrix, n * n, MPI_FLOAT, 0, comm);

    // Determine the block size for each process
    int block_size = n / size;

    // Allocate buffer for sending and receiving
    float* send_buffer = malloc(block_size * n * sizeof(float));
    float* recv_buffer = malloc(block_size * n * sizeof(float));
    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate send/recv buffers.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Prepare send buffer: each process sends its block of rows
    for(int i = 0; i < block_size; i++) {
        for(int j = 0; j < n; j++) {
            send_buffer[i * n + j] = flat_matrix[(rank * block_size + i) * n + j];
        }
    }

    // Measure the start time
    double start_time = MPI_Wtime();

    // Perform all-to-all communication
    MPI_Alltoall(send_buffer, block_size * n / size, MPI_FLOAT,
                 recv_buffer, block_size * n / size, MPI_FLOAT, comm);

    // Measure the end time
    double end_time = MPI_Wtime();
    *time = (long double)(end_time - start_time);

    // Allocate memory for the transposed flat matrix on all processes
    transposed_flat = malloc(n * n * sizeof(float));
    if (transposed_flat == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate memory for transposed_flat.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Rearrange received blocks into the transposed flat matrix
    for(int p = 0; p < size; p++) {
        for(int i = 0; i < block_size; i++) {
            for(int j = 0; j < block_size; j++) {
                transposed_flat[j * n + p * block_size + i] = recv_buffer[p * block_size * n / size + i * (n / size) + j];
            }
        }
    }

    // Create the 2D transposed matrix
    float** transposed = create_2d_matrix(transposed_flat, n, n);
    if(rank == 0 && transposed == NULL) {
        fprintf(stderr, "Root Rank %d: Failed to create transposed matrix.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Free allocated buffers
    free(flat_matrix);
    free(send_buffer);
    free(recv_buffer);
    free(transposed_flat);

    // Debugging Print at End
    if (verbosity >= 2) {
        printf("Rank %d: alltoall_transpose_mpi completed successfully\n", rank);
        fflush(stdout);
    }

    return transposed;
}

float** block_cyclic_transpose_mpi(MPI_Comm comm, float** matrix, int n, int rank, int size, long double* time, int verbosity) {
    int dims[2], periods[2] = {0, 0}, coords[2];
    MPI_Comm grid_comm;
    
    // Determine grid dimensions (assuming sqrt(size) is integer)
    dims[0] = dims[1] = (int)sqrt(size);
    if(dims[0] * dims[1] != size) {
        if(rank == 0) {
            fprintf(stderr, "Number of processes must be a perfect square.\n");
        }
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Create Cartesian topology
    MPI_Cart_create(comm, 2, dims, periods, 0, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    // Define block size
    int block_size = n / dims[0]; // Assuming n is divisible by dims[0]

    // Allocate local block
    float* local_block = malloc(block_size * block_size * sizeof(float));
    if (local_block == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate local_block.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Only root process flattens the matrix
    float* flat_matrix = NULL;
    if(rank == 0) {
        flat_matrix = flatten_matrix(matrix, n);
        if (flat_matrix == NULL) {
            fprintf(stderr, "Rank %d: Failed to flatten the matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Scatter blocks to all processes
    MPI_Datatype block_type, block_type_resized;
    MPI_Type_create_subarray(2, (int[]){n, n}, (int[]){block_size, block_size},
                            (int[]){coords[0] * block_size, coords[1] * block_size},
                            MPI_ORDER_C, MPI_FLOAT, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(float), &block_type_resized);
    MPI_Type_commit(&block_type_resized);

    int send_counts[size];
    int displs[size];
    if(rank == 0) {
        for(int i = 0; i < size; i++) {
            displs[i] = i;
            send_counts[i] = 1;
        }
    }

    MPI_Scatterv(flat_matrix, send_counts, displs, block_type_resized,
                 local_block, block_size * block_size, MPI_FLOAT,
                 0, grid_comm);

    // Measure the start time
    double start_time = MPI_Wtime();

    // Transpose the local block
    for(int i = 0; i < block_size; i++) {
        for(int j = i+1; j < block_size; j++) {
            float temp = local_block[i * block_size + j];
            local_block[i * block_size + j] = local_block[j * block_size + i];
            local_block[j * block_size + i] = temp;
        }
    }

    // Create the transposed block communicator
    MPI_Comm transposed_grid_comm;
    MPI_Cart_create(comm, 2, dims, periods, 1, &transposed_grid_comm);
    MPI_Cart_coords(transposed_grid_comm, rank, 2, coords);

    // Gather the transposed blocks back to the root
    float* transposed_flat = NULL;
    if(rank == 0) {
        transposed_flat = malloc(n * n * sizeof(float));
        if (transposed_flat == NULL) {
            fprintf(stderr, "Root Rank %d: Failed to allocate transposed_flat.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    MPI_Gatherv(local_block, block_size * block_size, MPI_FLOAT,
                transposed_flat, send_counts, displs, block_type_resized,
                0, transposed_grid_comm);

    // Measure the end time
    double end_time = MPI_Wtime();
    *time = (long double)(end_time - start_time);

    // Create the transposed 2D matrix on root
    float** transposed = NULL;
    if(rank == 0) {
        transposed = create_2d_matrix(transposed_flat, n, n);
        if (transposed == NULL) {
            fprintf(stderr, "Root Rank %d: Failed to create transposed matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Cleanup
    free(local_block);
    if(rank == 0) free(flat_matrix);
    if(rank == 0) free(transposed_flat);
    MPI_Type_free(&block_type);
    MPI_Type_free(&block_type_resized);
    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&transposed_grid_comm);

    // Debugging Print at End
    if (verbosity >= 2) {
        printf("Rank %d: block_cyclic_transpose_mpi completed successfully\n", rank);
        fflush(stdout);
    }

    return transposed;
}

float** nonblocking_transpose_mpi(MPI_Comm comm, float** matrix, int n, int rank, int size, long double* time, int verbosity) {
    float* flat_matrix = NULL;
    float* transposed_flat = NULL;

    // Enhanced Debugging Print
    if (verbosity >= 2) {
        printf("Rank %d: Starting nonblocking_transpose_mpi\n", rank);
        fflush(stdout);
    }

    // Only root process flattens the matrix
    if(rank == 0) {
        flat_matrix = flatten_matrix(matrix, n);
        if (flat_matrix == NULL) {
            fprintf(stderr, "Rank %d: Failed to flatten the matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Broadcast the flattened matrix size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);

    // Determine the number of rows per process
    int rows_per_proc = n / size;
    // No remainder due to n divisible by size and both being powers of two

    // Allocate memory for the local chunk
    float* local_matrix = malloc(rows_per_proc * n * sizeof(float));
    if (local_matrix == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate local_matrix.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Non-blocking scatter
    MPI_Request scatter_req;
    MPI_Iscatter(flat_matrix, rows_per_proc * n, MPI_FLOAT,
                local_matrix, rows_per_proc * n, MPI_FLOAT,
                0, comm, &scatter_req);

    // Start computation (if any pre-processing is needed)
    // For simplicity, we assume computation starts after initiating scatter

    // Wait for scatter to complete
    MPI_Wait(&scatter_req, MPI_STATUS_IGNORE);

    // Allocate memory for the local transposed chunk
    float* local_transposed = malloc(rows_per_proc * n * sizeof(float));
    if (local_transposed == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate local_transposed.\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Start non-blocking computation: transpose local chunk
    // Note: Actual overlapping depends on the hardware and MPI library
    MPI_Request transpose_req;
    MPI_Request compute_req;
    // For demonstration, we'll simulate overlapping by transposing in a separate thread
    // However, in pure MPI C, true overlapping would require pthreads or similar
    // Here, we'll proceed with the standard approach for simplicity

    // Measure the start time
    double start_time = MPI_Wtime();

    // Perform local transpose
    for(int i = 0; i < rows_per_proc; i++) {
        for(int j = 0; j < n; j++) {
            local_transposed[j * rows_per_proc + i] = local_matrix[i * n + j];
        }
    }

    // Measure the end time
    double end_time = MPI_Wtime();
    *time = (long double)(end_time - start_time);

    // Allocate memory for the transposed flat matrix on root
    if(rank == 0) {
        transposed_flat = malloc(n * n * sizeof(float));
        if (transposed_flat == NULL) {
            fprintf(stderr, "Root Rank %d: Failed to allocate transposed_flat.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Non-blocking gather
    MPI_Request gather_req;
    MPI_Igather(local_transposed, rows_per_proc * n, MPI_FLOAT,
                transposed_flat, rows_per_proc * n, MPI_FLOAT,
                0, comm, &gather_req);

    // Continue with other computations if needed while gather is in progress

    // Wait for gather to complete
    MPI_Wait(&gather_req, MPI_STATUS_IGNORE);

    // Root process reconstructs the transposed matrix
    float** transposed = NULL;
    if(rank == 0) {
        transposed = create_2d_matrix(transposed_flat, n, n);
        if (transposed == NULL) {
            fprintf(stderr, "Root Rank %d: Failed to create transposed matrix.\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
        // Free flattened arrays
        free(flat_matrix);
        free(transposed_flat);
    }

    // Cleanup
    free(local_matrix);
    free(local_transposed);

    // Debugging Print at End
    if (verbosity >= 2) {
        printf("Rank %d: nonblocking_transpose_mpi completed successfully\n", rank);
        fflush(stdout);
    }

    return transposed;
}
