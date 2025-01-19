import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

def plot_speedup(df, output_dir):
    """
    Plots a 2D scatter plot comparing the speedup of the best runtimes of each function
    relative to the sequential function against the matrix size.

    Parameters:
    - df: A DataFrame containing the benchmark results.
    - output_dir: Directory where the plot will be saved.
    """
    # Filter out unwanted function names
    unwanted_funcs = ['is_symmetric_sequential', 'is_symmetric_implicit', 'is_symmetric_omp']
    df_filtered = df[~df['func_name'].isin(unwanted_funcs)]

    # Get the best runtime for each function and matrix size
    idx = df_filtered.groupby(['matrix_dimension', 'func_name'])['time'].idxmin()
    best_runtimes = df_filtered.loc[idx].reset_index(drop=True)

    # Get the best runtime for the sequential function for each matrix size
    sequential_runtimes = df_filtered[df_filtered['func_name'] == 'transpose_sequential'].groupby('matrix_dimension')['time'].min().reset_index()
    sequential_runtimes.rename(columns={'time': 'time_sequential'}, inplace=True)

    # Merge the best runtimes with the sequential runtimes
    merged = pd.merge(best_runtimes, sequential_runtimes, on='matrix_dimension')

    # Calculate the speedup
    merged['speedup'] = merged['time_sequential'] / merged['time']

    # Define markers for different functions
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    func_names = merged['func_name'].unique()
    marker_dict = {func_name: markers[i % len(markers)] for i, func_name in enumerate(func_names)}

    # Plot the speedup
    plt.figure(figsize=(10, 6))
    for func_name in func_names:
        subset = merged[merged['func_name'] == func_name]
        plt.scatter(subset['matrix_dimension'], subset['speedup'],
                    label=func_name, marker=marker_dict[func_name])

    plt.xscale('log', base=2)
    plt.xlabel('Matrix Dimension (log scale)')
    plt.ylabel('Speedup')
    plt.title('Speedup of Best Runtimes Relative to Sequential Function')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Set x-tick labels to the encountered matrix dimensions
    plt.xticks(ticks=sorted(merged['matrix_dimension'].unique()),
               labels=sorted(merged['matrix_dimension'].unique()), rotation=45)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'speedup_plot.png'), bbox_inches='tight')
    plt.close()

def plot_strong_scaling(df, output_dir):
    """
    Plots strong scaling analysis showing how execution time decreases with increasing number of threads
    for a fixed matrix size.

    Parameters:
    - df: A DataFrame containing the benchmark results.
    - output_dir: Directory where the plot will be saved.
    """
    # Define a fixed matrix size (use the largest matrix size for comprehensive analysis)
    fixed_matrix_size = df['matrix_dimension'].max()

    # Filter dataframe for the fixed matrix size
    strong_df = df[df['matrix_dimension'] == fixed_matrix_size]

    # Exclude sequential runtime for parallel comparisons
    parallel_df = strong_df[strong_df['func_name'] != 'transpose_sequential']

    if parallel_df.empty:
        print(f"No parallel implementations found for matrix size {fixed_matrix_size}. Skipping strong scaling plot.")
        return

    # Get the best runtime for each function based on threads
    idx = parallel_df.groupby(['func_name', 'threads'])['time'].idxmin()
    best_parallel = parallel_df.loc[idx].reset_index(drop=True)

    # Use parallel implementation with threads=1 as the baseline for each function
    baseline = best_parallel[best_parallel['threads'] == 1].rename(columns={'time': 'time_baseline'})
    baseline = baseline[['func_name', 'time_baseline']]

    # Merge baseline with best_parallel
    merged = pd.merge(best_parallel, baseline, on='func_name')

    # Calculate speedup and efficiency
    merged['speedup'] = merged['time_baseline'] / merged['time']
    merged['efficiency'] = merged['speedup'] / merged['threads']

    # Plot Execution Time vs Number of Threads
    plt.figure(figsize=(10, 6))
    for func_name in merged['func_name'].unique():
        subset = merged[merged['func_name'] == func_name]
        plt.plot(subset['threads'], subset['time'], marker='o', label=func_name)

    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Number of Threads (log scale)')
    plt.ylabel('Execution Time (ms, log scale)')
    plt.title(f'Strong Scaling for Matrix Dimension {fixed_matrix_size}')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xticks(ticks=sorted(merged['threads'].unique()),
               labels=sorted(merged['threads'].unique()), rotation=45)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'strong_scaling_execution_time.png'), bbox_inches='tight')
    plt.close()

    # Plot Speedup vs Number of Threads
    plt.figure(figsize=(10, 6))
    for func_name in merged['func_name'].unique():
        subset = merged[merged['func_name'] == func_name]
        plt.plot(subset['threads'], subset['speedup'], marker='o', label=func_name)

    plt.xscale('log', base=2)
    plt.xlabel('Number of Threads (log scale)')
    plt.ylabel('Speedup')
    plt.title(f'Strong Scaling Speedup for Matrix Dimension {fixed_matrix_size}')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xticks(ticks=sorted(merged['threads'].unique()),
               labels=sorted(merged['threads'].unique()), rotation=45)

    plt.savefig(os.path.join(output_dir, 'strong_scaling_speedup.png'), bbox_inches='tight')
    plt.close()

    # Plot Efficiency vs Number of Threads
    plt.figure(figsize=(10, 6))
    for func_name in merged['func_name'].unique():
        subset = merged[merged['func_name'] == func_name]
        plt.plot(subset['threads'], subset['efficiency'], marker='o', label=func_name)

    plt.xscale('log', base=2)
    plt.xlabel('Number of Threads (log scale)')
    plt.ylabel('Efficiency')
    plt.title(f'Strong Scaling Efficiency for Matrix Dimension {fixed_matrix_size}')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xticks(ticks=sorted(merged['threads'].unique()),
               labels=sorted(merged['threads'].unique()), rotation=45)

    plt.savefig(os.path.join(output_dir, 'strong_scaling_efficiency.png'), bbox_inches='tight')
    plt.close()

def plot_weak_scaling(df, output_dir):
    """
    Plots weak scaling analysis showing how execution time varies with increasing number of threads
    when the problem size per thread is fixed.

    Parameters:
    - df: A DataFrame containing the benchmark results.
    - output_dir: Directory where the plot will be saved.
    """
    # Assuming weak scaling: matrix_dimension ∝ threads
    # Since matrix_dimension and threads scale exponentially with base 2, we can pair them accordingly

    # Calculate the problem size per thread
    df_sorted = df.sort_values(by=['threads', 'matrix_dimension']).copy()
    df_sorted['problem_size_per_thread'] = df_sorted['matrix_dimension'] / df_sorted['threads']

    # Determine the most common problem_size_per_thread
    problem_size_fixed = df_sorted['problem_size_per_thread'].mode()[0]

    # Allow a small tolerance for floating point arithmetic
    tolerance = 0.1
    weak_scaling_df = df_sorted[np.abs(df_sorted['problem_size_per_thread'] - problem_size_fixed) < tolerance]

    if weak_scaling_df.empty:
        print("No suitable data found for weak scaling analysis.")
        return

    # Exclude sequential function
    parallel_df = weak_scaling_df[~weak_scaling_df['func_name'].isin(['transpose_sequential'])]

    if parallel_df.empty:
        print("No parallel implementations found for weak scaling analysis.")
        return

    # Aggregate execution time by function and threads (mean)
    best_parallel = parallel_df.groupby(['func_name', 'threads'])['time'].mean().reset_index()

    # Plot Execution Time vs Number of Threads
    plt.figure(figsize=(10, 6))
    for func_name in best_parallel['func_name'].unique():
        subset = best_parallel[best_parallel['func_name'] == func_name]
        plt.plot(subset['threads'], subset['time'], marker='o', label=func_name)

    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Number of Threads (log scale)')
    plt.ylabel('Execution Time (ms, log scale)')
    plt.title('Weak Scaling Execution Time')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xticks(ticks=sorted(best_parallel['threads'].unique()),
               labels=sorted(best_parallel['threads'].unique()), rotation=45)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'weak_scaling_execution_time.png'), bbox_inches='tight')
    plt.close()

def plot_weak_scaling_additional_metrics(df, output_dir):
    """
    Plots additional metrics (speedup and scalability) for weak scaling.

    Note: Weak scaling speedup and scalability metrics are not standard and can be misleading.
    It's recommended to focus on execution time for weak scaling.
    """
    # Weak scaling speedup and scalability are not meaningful; hence, we skip these plots.
    print("Weak scaling speedup and scalability plots are not generated as they are not meaningful metrics.")

def plot_2d_execution_time(df, output_dir):
    """
    Plots 2D execution time graphs for specified functions.

    Parameters:
    - df: A DataFrame containing the benchmark results.
    - output_dir: Directory where the plots will be saved.
    """
    functions = ['transpose_implicit', 'transpose_sequential', 'transpose_mpi']

    for func in functions:
        subset = df[df['func_name'] == func]

        # Aggregate data to avoid multiple points for the same matrix_dimension
        aggregated = subset.groupby('matrix_dimension')['time'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.plot(aggregated['matrix_dimension'], aggregated['time'],
                 marker='o', label=func)

        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.xlabel('Matrix Dimension (log scale)')
        plt.ylabel('Time (ms, log scale)')
        plt.title(f'Execution Time vs Matrix Dimension for {func}')
        plt.legend()
        plt.grid(True, which="both", ls="--", linewidth=0.5)

        # Set x-tick labels to the encountered matrix dimensions
        plt.xticks(ticks=sorted(aggregated['matrix_dimension'].unique()),
                   labels=sorted(aggregated['matrix_dimension'].unique()), rotation=45)

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{func}_2d_plot.png'), bbox_inches='tight')
        plt.close()

    # Compare speedup between 'transpose_implicit', 'transpose_sequential', and 'transpose_mpi'
    implicit = df[df['func_name'] == 'transpose_implicit']
    sequential = df[df['func_name'] == 'transpose_sequential']
    mpi = df[df['func_name'] == 'transpose_mpi']

    # Aggregate data to avoid multiple entries for the same matrix_dimension
    implicit_avg = implicit.groupby('matrix_dimension')['time'].mean().reset_index()
    sequential_avg = sequential.groupby('matrix_dimension')['time'].mean().reset_index()
    mpi_avg = mpi.groupby('matrix_dimension')['time'].mean().reset_index()

    # Merge the dataframes
    merged = pd.merge(implicit_avg, sequential_avg, on='matrix_dimension', suffixes=('_implicit', '_sequential'))
    merged = pd.merge(merged, mpi_avg, on='matrix_dimension', suffixes=('', '_mpi'))

    # Calculate speedups
    merged['speedup_implicit'] = merged['time_sequential'] / merged['time_implicit']
    merged['speedup_mpi'] = merged['time_sequential'] / merged['time']

    plt.figure(figsize=(10, 6))
    plt.plot(merged['matrix_dimension'], merged['speedup_implicit'],
             marker='o', label='Speedup (Implicit vs Sequential)')
    plt.plot(merged['matrix_dimension'], merged['speedup_mpi'],
             marker='s', label='Speedup (MPI vs Sequential)')

    plt.xscale('log', base=2)
    plt.xlabel('Matrix Dimension (log scale)')
    plt.ylabel('Speedup')
    plt.title('Speedup of Implicit and MPI vs Sequential')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Set x-tick labels to the encountered matrix dimensions
    plt.xticks(ticks=sorted(merged['matrix_dimension'].unique()),
               labels=sorted(merged['matrix_dimension'].unique()), rotation=45)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'implicit_vs_sequential_vs_mpi_speedup.png'), bbox_inches='tight')
    plt.close()

def plot_mpi_2d_execution_time(df, output_dir):
    """
    Plots 2D execution time graphs for MPI-specific functions.

    Parameters:
    - df: A DataFrame containing the benchmark results.
    - output_dir: Directory where the plots will be saved.
    """
    mpi_functions = [
        'is_symmetric_mpi',
        'transpose_mpi',
        'alltoall_transpose_mpi',
        'block_cyclic_transpose_mpi',
        'nonblocking_transpose_mpi'
    ]

    for func in mpi_functions:
        subset = df[df['func_name'] == func]

        # Aggregate data to avoid multiple points for the same matrix_dimension
        aggregated = subset.groupby('matrix_dimension')['time'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.plot(aggregated['matrix_dimension'], aggregated['time'],
                 marker='o', label=func)

        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.xlabel('Matrix Dimension (log scale)')
        plt.ylabel('Time (ms, log scale)')
        plt.title(f'Execution Time vs Matrix Dimension for {func}')
        plt.legend()
        plt.grid(True, which="both", ls="--", linewidth=0.5)

        # Set x-tick labels to the encountered matrix dimensions
        plt.xticks(ticks=sorted(aggregated['matrix_dimension'].unique()),
                   labels=sorted(aggregated['matrix_dimension'].unique()), rotation=45)

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{func}_2d_plot.png'), bbox_inches='tight')
        plt.close()

def plot_additional_metrics(df, output_dir):
    """
    Plots additional useful metrics such as efficiency for parallel implementations.

    Parameters:
    - df: A DataFrame containing the benchmark results.
    - output_dir: Directory where the plots will be saved.
    """
    # Define parallel functions (excluding sequential)
    parallel_funcs = df[~df['func_name'].isin(['transpose_sequential'])]['func_name'].unique()

    for func in parallel_funcs:
        subset = df[df['func_name'] == func]

        # Use parallel implementation with threads=1 as the baseline
        baseline_df = subset[subset['threads'] == 1]
        if baseline_df.empty:
            print(f"No baseline (threads=1) found for function {func}. Skipping efficiency plot.")
            continue
        baseline_time = baseline_df['time'].min()

        # Aggregate data by threads to compute speedup and efficiency
        aggregated = subset.groupby('threads')['time'].min().reset_index()
        aggregated['speedup'] = baseline_time / aggregated['time']
        aggregated['efficiency'] = aggregated['speedup'] / aggregated['threads']

        # Plot Efficiency vs Number of Threads
        plt.figure(figsize=(10, 6))
        plt.plot(aggregated['threads'], aggregated['efficiency'],
                 marker='o', label='Efficiency')

        plt.xscale('log', base=2)
        plt.xlabel('Number of Threads (log scale)')
        plt.ylabel('Efficiency')
        plt.title(f'Efficiency vs Number of Threads for {func}')
        plt.legend()
        plt.grid(True, which="both", ls="--", linewidth=0.5)

        plt.xticks(ticks=sorted(aggregated['threads'].unique()),
                   labels=sorted(aggregated['threads'].unique()), rotation=45)

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{func}_efficiency.png'), bbox_inches='tight')
        plt.close()

def main():
    # Read the CSV file
    try:
        df = pd.read_csv('benchmark_results.csv')
    except FileNotFoundError:
        print("The file 'benchmark_results.csv' was not found.")
        return

    # Ensure that data types are correct
    df['threads'] = df['threads'].astype(int)
    df['matrix_dimension'] = df['matrix_dimension'].astype(int)
    df['block_size'] = df['block_size'].astype(int)
    df['time'] = df['time'].astype(float)
    df['func_name'] = df['func_name'].astype(str)

    # Define base output directories
    base_output_dir = 'plots'
    speedup_dir = os.path.join(base_output_dir, 'speedup')
    strong_scaling_dir = os.path.join(base_output_dir, 'strong_scaling')
    weak_scaling_dir = os.path.join(base_output_dir, 'weak_scaling')
    execution_time_2d_dir = os.path.join(base_output_dir, '2d_execution_time')
    mpi_2d_dir = os.path.join(base_output_dir, 'mpi_2d_execution_time')
    additional_metrics_dir = os.path.join(base_output_dir, 'additional_metrics')

    # Plot Speedup
    plot_speedup(df, speedup_dir)

    # Plot 2D Execution Time for specific functions
    plot_2d_execution_time(df, execution_time_2d_dir)

    # Plot MPI-specific 2D Execution Time
    plot_mpi_2d_execution_time(df, mpi_2d_dir)

    # Plot Strong Scaling
    plot_strong_scaling(df, strong_scaling_dir)

    # Plot Weak Scaling
    plot_weak_scaling(df, weak_scaling_dir)

    # Note: Weak scaling speedup and scalability metrics are not meaningful and hence are not plotted
    # plot_weak_scaling_additional_metrics(df, weak_scaling_dir)

    # Plot Additional Metrics (Efficiency)
    plot_additional_metrics(df, additional_metrics_dir)

    print("All plots have been generated and saved in the 'plots' directory.")

if __name__ == "__main__":
    main()
