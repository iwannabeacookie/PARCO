import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

def plot_3d_graph(group, data):
    """
    Plots a 3D surface graph for the given group of data.
    
    Parameters:
    - group: A tuple containing the group identifiers (matrix_dimension, func_name).
    - data: A DataFrame containing the subset of data for the group.
    """
    # Unpack group identifiers
    matrix_dimension, func_name = group
    
    # Pivot the data to create a matrix for Z-axis (time)
    # Use pivot_table to handle duplicate (threads, block_size) pairs by taking the mean time
    pivot_table = data.pivot_table(index='threads', columns='block_size', values='time', aggfunc='mean')
    
    # Ensure that the pivot_table does not contain any missing values
    # if pivot_table.isnull().values.any():
    #     # Handle missing data. Here, we'll fill missing values with the mean time
    #     pivot_table = pivot_table.fillna(pivot_table.mean().mean())
    #     print(f"Missing data filled with mean time for group: {group}")
    
    # Extract sorted unique threads and block_sizes
    threads = np.sort(pivot_table.index.values)
    block_sizes = np.sort(pivot_table.columns.values)
    
    # Create meshgrid for X and Y
    X, Y = np.meshgrid(block_sizes, threads)
    
    # Extract Z values
    Z = pivot_table.values
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Wirefram graph does not seem to provide a good visualization
    # surf = ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm, linewidth=0.5)
    
    # Add a color bar for reference
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Time (ms)')
    
    # Set plot titles and labels
    ax.set_title(f'{matrix_dimension} - {func_name}', fontsize=15)
    ax.set_xlabel('Block Size', fontsize=12)
    ax.set_ylabel('Threads', fontsize=12)
    ax.set_zlabel('Time (ms)', fontsize=12)

    ax.invert_yaxis()
    
    # Optional: Invert Y-axis if desired
    # ax.invert_yaxis()
    
    # Customize the view angle for better visualization
    ax.view_init(elev=30, azim=225)
    
    # Define the output directory and save the plot
    output_dir = os.path.join('plots', str(matrix_dimension), func_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '3d_plot.png'), bbox_inches='tight')
    plt.close()

def plot_speedup(df):
    """
    Plots a 2D scatter plot comparing the speedup of the best runtimes of each function
    relative to the sequential function against the matrix size.
    
    Parameters:
    - df: A DataFrame containing the benchmark results.
    """
    # Filter out unwanted function names
    df = df.query("func_name not in ['is_symmetric_sequential', 'is_symmetric_implicit', 'is_symmetric_omp']")

    # Get the best runtime for each function and matrix size
    idx = df.groupby(['matrix_dimension', 'func_name'])['time'].idxmin()
    best_runtimes = df.loc[idx].reset_index(drop=True)

    # Get the best runtime for the sequential function for each matrix size
    sequential_runtimes = df[df['func_name'] == 'transpose_sequential'].groupby('matrix_dimension')['time'].min().reset_index()
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
        plt.scatter(subset['matrix_dimension'], subset['speedup'], label=func_name, marker=marker_dict[func_name])
        # for _, row in subset.iterrows():
        #     plt.annotate(f"BS: {row['block_size']}, T: {row['threads']}", 
        #                  (row['matrix_dimension'], row['speedup']),
        #                  textcoords="offset points", xytext=(0,10), ha='center')

    plt.xscale('log')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Speedup')
    plt.title('Speedup of Best Runtimes Relative to Sequential Function')
    plt.legend()
    plt.grid(True)
    
    # Set x-tick labels to the encountered matrix dimensions
    plt.xticks(ticks=merged['matrix_dimension'].unique(), labels=merged['matrix_dimension'].unique(), rotation=45)
    
    plt.savefig('speedup_plot.png', bbox_inches='tight')
    plt.close()

def plot_2d(df):
    """
    Plots 2D graphs for 'transpose_implicit', 'transpose_sequential', and 'transpose_mpi' functions,
    and compares speedup between them.
    
    Parameters:
    - df: A DataFrame containing the benchmark results.
    """
    functions = ['transpose_implicit', 'transpose_sequential', 'transpose_mpi']
    
    for func in functions:
        subset = df[df['func_name'] == func]
        
        plt.figure(figsize=(10, 6))
        plt.plot(subset['matrix_dimension'], subset['time'], marker='o', label=func)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Matrix Dimension')
        plt.ylabel('Time (ms)')
        plt.title(f'Execution Time vs Matrix Dimension for {func}')
        plt.legend()
        plt.grid(True)
        
        # Set x-tick labels to the encountered matrix dimensions
        plt.xticks(ticks=subset['matrix_dimension'].unique(), labels=subset['matrix_dimension'].unique(), rotation=45)
        
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{func}_2d_plot.png'), bbox_inches='tight')
        plt.close()
    
    # Compare speedup between 'transpose_implicit', 'transpose_sequential', and 'transpose_mpi'
    implicit = df[df['func_name'] == 'transpose_implicit']
    sequential = df[df['func_name'] == 'transpose_sequential']
    mpi = df[df['func_name'] == 'transpose_mpi']
    
    merged = pd.merge(implicit, sequential, on='matrix_dimension', suffixes=('_implicit', '_sequential'))
    merged = pd.merge(merged, mpi, on='matrix_dimension', suffixes=('', '_mpi'))
    merged['speedup_implicit'] = merged['time_sequential'] / merged['time_implicit']
    merged['speedup_mpi'] = merged['time_sequential'] / merged['time']
    
    plt.figure(figsize=(10, 6))
    plt.plot(merged['matrix_dimension'], merged['speedup_implicit'], marker='o', label='Speedup (Implicit vs Sequential)')
    plt.plot(merged['matrix_dimension'], merged['speedup_mpi'], marker='s', label='Speedup (MPI vs Sequential)')
    
    plt.xscale('log')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Speedup')
    plt.title('Speedup of Implicit and MPI vs Sequential')
    plt.legend()
    plt.grid(True)
    
    # Set x-tick labels to the encountered matrix dimensions
    plt.xticks(ticks=merged['matrix_dimension'].unique(), labels=merged['matrix_dimension'].unique(), rotation=45)
    
    plt.savefig(os.path.join(output_dir, 'implicit_vs_sequential_vs_mpi_speedup.png'), bbox_inches='tight')
    plt.close()

def main():
    # Read the CSV file
    df = pd.read_csv('benchmark_results.csv')

    plot_speedup(df)
    plot_2d(df)
    
    # Filter out unwanted function names
    df = df.query("func_name not in ['is_symmetric_sequential', 'is_symmetric_implicit', 'is_symmetric_omp']")
    
    # Group the data by 'matrix_dimension' and 'func_name'
    grouped = df.groupby(['matrix_dimension', 'func_name'])
    
    # Iterate over each group and plot
    for group, grouped_df in grouped:
        print(f"Processing group: {group} with {len(grouped_df)} records")
        plot_3d_graph(group, grouped_df)

if __name__ == "__main__":
    main()
