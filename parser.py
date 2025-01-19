import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle

def plot_speedup(df, output_dir):
    """
    Plots a 2D line plot comparing the speedup of each parallel function relative to the sequential function
    against the matrix size.

    Parameters:
    - df: A DataFrame containing the benchmark results.
    - output_dir: Directory where the plot will be saved.
    """
    # Filter out unwanted function names
    unwanted_funcs = ['is_symmetric_sequential', 'is_symmetric_implicit', 'is_symmetric_omp']
    df_filtered = df[~df['func_name'].isin(unwanted_funcs)]

    # Get the best (minimum) runtime for each function and matrix size
    idx = df_filtered.groupby(['matrix_dimension', 'func_name'])['time'].idxmin()
    best_runtimes = df_filtered.loc[idx].reset_index(drop=True)

    # Get the best runtime for the sequential function for each matrix size
    sequential_runtimes = df_filtered[df_filtered['func_name'] == 'transpose_sequential'] \
                            .groupby('matrix_dimension')['time'].min().reset_index()
    sequential_runtimes.rename(columns={'time': 'time_sequential'}, inplace=True)

    # Merge the best runtimes with the sequential runtimes
    merged = pd.merge(best_runtimes, sequential_runtimes, on='matrix_dimension')

    # Handle potential zero or undefined runtimes by replacing them with a very small number
    merged['time'] = merged['time'].replace(0, merged['time'][merged['time'] > 0].min())
    merged['time_sequential'] = merged['time_sequential'].replace(0, merged['time_sequential'][merged['time_sequential'] > 0].min())

    # Calculate the speedup
    merged['speedup'] = merged['time_sequential'] / merged['time']

    # Define a color cycle with enough distinct colors using tab20 colormap
    color_cycle = cycle(cm.tab20.colors)  # tab20 has 20 distinct colors

    plt.figure(figsize=(12, 8))

    for func_name, group in merged.groupby('func_name'):
        color = next(color_cycle)
        # Sort by matrix_dimension to ensure lines connect correctly
        group_sorted = group.sort_values('matrix_dimension')
        plt.plot(group_sorted['matrix_dimension'], group_sorted['speedup'],
                 marker='o', label=func_name, color=color, linestyle='-')

    plt.xscale('log', base=2)
    plt.xlabel('Matrix Dimension (log scale)')
    plt.ylabel('Speedup')
    plt.title('Speedup of Parallel Functions Relative to Sequential Function')
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
    for a fixed matrix size, excluding the sequential implementation.

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

    # Get the best (minimum) runtime for each function based on threads
    idx = parallel_df.groupby(['func_name', 'threads'])['time'].idxmin()
    best_parallel = parallel_df.loc[idx].reset_index(drop=True)

    # Assume that for strong scaling, threads=1 within parallel implementations are the baseline
    baseline = best_parallel[best_parallel['threads'] == 1][['func_name', 'time']]
    baseline.rename(columns={'time': 'time_baseline'}, inplace=True)

    # Merge baseline with best_parallel
    merged = pd.merge(best_parallel, baseline, on='func_name', how='left')

    # Replace any missing baseline times with the smallest non-zero time in the dataset
    merged['time_baseline'] = merged['time_baseline'].fillna(merged['time_baseline'].replace(0, merged['time_baseline'][merged['time_baseline'] > 0].min()))

    # Calculate speedup and efficiency
    merged['speedup'] = merged['time_baseline'] / merged['time']
    merged['efficiency'] = merged['speedup'] / merged['threads']

    # Handle potential zero or undefined efficieny
    merged['efficiency'] = merged['efficiency'].replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=['speedup', 'efficiency'])

    # Define a color cycle with enough distinct colors using tab20 colormap
    color_cycle = cycle(cm.tab20.colors)

    # Plot Execution Time vs Number of Threads
    plt.figure(figsize=(12, 8))
    for func_name, group in merged.groupby('func_name'):
        color = next(color_cycle)
        group_sorted = group.sort_values('threads')
        plt.plot(group_sorted['threads'], group_sorted['time'],
                 marker='o', label=func_name, color=color, linestyle='-')

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

    # Reset color cycle for the next plot to maintain distinct colors
    color_cycle = cycle(cm.tab20.colors)

    # Plot Speedup vs Number of Threads
    plt.figure(figsize=(12, 8))
    for func_name, group in merged.groupby('func_name'):
        color = next(color_cycle)
        group_sorted = group.sort_values('threads')
        plt.plot(group_sorted['threads'], group_sorted['speedup'],
                 marker='o', label=func_name, color=color, linestyle='-')

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

    # Reset color cycle for the next plot to maintain distinct colors
    color_cycle = cycle(cm.tab20.colors)

    # Plot Efficiency vs Number of Threads
    plt.figure(figsize=(12, 8))
    for func_name, group in merged.groupby('func_name'):
        color = next(color_cycle)
        group_sorted = group.sort_values('threads')
        plt.plot(group_sorted['threads'], group_sorted['efficiency'],
                 marker='o', label=func_name, color=color, linestyle='-')

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

    # Get the best (minimum) runtime for each function and thread count
    idx = parallel_df.groupby(['func_name', 'threads'])['time'].idxmin()
    best_parallel = parallel_df.loc[idx].reset_index(drop=True)

    # Handle potential zero runtimes by replacing them with the smallest non-zero time in the dataset
    best_parallel['time'] = best_parallel['time'].replace(0, best_parallel['time'][best_parallel['time'] > 0].min())

    # Define a color cycle with enough distinct colors using tab20 colormap
    color_cycle = cycle(cm.tab20.colors)

    # Plot Execution Time vs Number of Threads
    plt.figure(figsize=(12, 8))
    for func_name, group in best_parallel.groupby('func_name'):
        color = next(color_cycle)
        group_sorted = group.sort_values('threads')
        plt.plot(group_sorted['threads'], group_sorted['time'],
                 marker='o', label=func_name, color=color, linestyle='-')

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

def plot_2d_execution_time(df, output_dir):
    """
    Plots 2D execution time graphs for specified functions.

    Parameters:
    - df: A DataFrame containing the benchmark results.
    - output_dir: Directory where the plots will be saved.
    """
    functions = ['transpose_implicit', 'transpose_sequential', 'transpose_mpi']

    # Define a color cycle with enough distinct colors
    color_cycle = cycle(cm.tab20.colors)

    for func in functions:
        subset = df[df['func_name'] == func]

        if subset.empty:
            print(f"No data available for function {func}. Skipping plot.")
            continue

        # Aggregate data to avoid multiple points for the same matrix_dimension
        aggregated = subset.groupby('matrix_dimension')['time'].min().reset_index()

        plt.figure(figsize=(12, 8))
        color = next(color_cycle)
        plt.plot(aggregated['matrix_dimension'], aggregated['time'],
                 marker='o', label=func, color=color, linestyle='-')

        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.xlabel('Matrix Dimension (log scale)')
        plt.ylabel('Execution Time (ms, log scale)')
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

    if sequential.empty:
        print("No sequential data available for speedup comparison. Skipping speedup plot.")
        return

    # Aggregate data to avoid multiple entries for the same matrix_dimension
    implicit_avg = implicit.groupby('matrix_dimension')['time'].min().reset_index()
    sequential_avg = sequential.groupby('matrix_dimension')['time'].min().reset_index()
    mpi_avg = mpi.groupby('matrix_dimension')['time'].min().reset_index()

    # Merge the dataframes
    merged = pd.merge(implicit_avg, sequential_avg, on='matrix_dimension', how='left', suffixes=('_implicit', '_sequential'))
    merged = pd.merge(merged, mpi_avg, on='matrix_dimension', how='left', suffixes=('', '_mpi'))

    # Replace zero runtimes with the smallest non-zero time to avoid division issues
    merged['time_implicit'] = merged['time_implicit'].replace(0, merged['time_implicit'][merged['time_implicit'] > 0].min())
    merged['time'] = merged['time'].replace(0, merged['time'][merged['time'] > 0].min())

    # Calculate speedups
    merged['speedup_implicit'] = merged['time_sequential'] / merged['time_implicit']
    merged['speedup_mpi'] = merged['time_sequential'] / merged['time']

    # Define a color cycle with enough distinct colors
    color_cycle_speedup = cycle(cm.tab20.colors)

    plt.figure(figsize=(12, 8))
    if 'speedup_implicit' in merged.columns:
        plt.plot(merged['matrix_dimension'], merged['speedup_implicit'],
                 marker='o', label='Speedup (Implicit vs Sequential)', color=next(color_cycle_speedup), linestyle='-')
    if 'speedup_mpi' in merged.columns:
        plt.plot(merged['matrix_dimension'], merged['speedup_mpi'],
                 marker='s', label='Speedup (MPI vs Sequential)', color=next(color_cycle_speedup), linestyle='-')

    plt.xscale('log', base=2)
    plt.xlabel('Matrix Dimension (log scale)')
    plt.ylabel('Speedup')
    plt.title('Speedup of Implicit and MPI vs Sequential')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Set x-tick labels
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

    # Define a color cycle with enough distinct colors
    color_cycle = cycle(cm.tab20.colors)

    for func in mpi_functions:
        subset = df[df['func_name'] == func]

        if subset.empty:
            print(f"No data available for MPI function {func}. Skipping plot.")
            continue

        # Aggregate data to avoid multiple points for the same matrix_dimension
        aggregated = subset.groupby('matrix_dimension')['time'].min().reset_index()

        plt.figure(figsize=(12, 8))
        color = next(color_cycle)
        plt.plot(aggregated['matrix_dimension'], aggregated['time'],
                 marker='o', label=func, color=color, linestyle='-')

        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.xlabel('Matrix Dimension (log scale)')
        plt.ylabel('Execution Time (ms, log scale)')
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

    # Define a color cycle with enough distinct colors
    color_cycle = cycle(cm.tab20.colors)

    for func in parallel_funcs:
        subset = df[df['func_name'] == func]

        # Identify if the implementation is MPI or OMP based on func_name
        # Here, we assume 'mpi' in func_name indicates MPI implementation
        if 'mpi' in func.lower():
            # For MPI, use threads=1 as baseline (assuming threads represent processes here)
            baseline = subset[subset['threads'] == 1]
        else:
            # For OMP, use threads=1 as baseline
            baseline = subset[subset['threads'] == 1]

        if baseline.empty:
            print(f"No baseline (threads=1) found for function {func}. Skipping efficiency plot.")
            continue

        baseline_time = baseline['time'].min()

        # Aggregate by threads, taking the min time to represent best case
        aggregated = subset.groupby('threads')['time'].min().reset_index()

        # Handle potential zero runtimes by replacing them with the smallest non-zero time
        aggregated['time'] = aggregated['time'].replace(0, aggregated['time'][aggregated['time'] > 0].min())

        # Calculate speedup and efficiency
        aggregated['speedup'] = baseline_time / aggregated['time']
        aggregated['efficiency'] = aggregated['speedup'] / aggregated['threads']

        # Define a color cycle reset to ensure distinct colors
        color = next(color_cycle)

        # Plot Efficiency vs Number of Threads
        plt.figure(figsize=(12, 8))
        plt.plot(aggregated['threads'], aggregated['efficiency'],
                 marker='o', label='Efficiency', color=color, linestyle='-')

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

    # Replace zero runtime measurements with the smallest non-zero runtime to avoid plotting zeros
    df['time'] = df['time'].replace(0, np.nan)
    min_non_zero_time = df['time'].min()
    df['time'] = df['time'].fillna(min_non_zero_time)

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

    # Plot Weak Scaling Execution Time Only
    plot_weak_scaling(df, weak_scaling_dir)

    # Plot Additional Metrics (Efficiency)
    plot_additional_metrics(df, additional_metrics_dir)

    print("All plots have been generated and saved in the 'plots' directory.")

if __name__ == "__main__":
    main()
