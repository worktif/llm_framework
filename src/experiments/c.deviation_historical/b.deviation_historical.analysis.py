import os

import pandas as pd
from matplotlib import pyplot as plt

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Define output folders for plots and data
output_folder_plots = f"{current_dir}/experimental_data/plots/model/deviation_historical.analysis"
os.makedirs(output_folder_plots, exist_ok=True)

output_folder_data = f"{current_dir}/experimental_data/data/model/deviation_historical.analysis"
os.makedirs(output_folder_data, exist_ok=True)

# Define the folder containing the deviation data
data_folder = f"{current_dir}/experimental_data/data/model/deviation_historical"

def analyze_and_plot_deviation_by_window(folder_path):
    """
    Analyze historical context deviation data and plot mean deviation by window size.

    Parameters:
    - folder_path: Path to the folder containing CSV files with deviation data.

    Saves:
    - A plot showing mean deviation by context window size.
    - A CSV file with mean deviations for each window size.
    """
    # Dictionary to store deviation data grouped by window size
    deviations_by_window = {}

    # Iterate through all files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and not file_name.startswith('deviation_historical'):
            # Extract the window size from the file name (assumes format contains 'window_X')
            window_size = int(file_name.split('window_')[-1].split('.')[0])
            file_path = os.path.join(folder_path, file_name)

            # Load data from the CSV file
            data = pd.read_csv(file_path)

            # Add deviations to the dictionary under the corresponding window size
            if window_size not in deviations_by_window:
                deviations_by_window[window_size] = []
            deviations_by_window[window_size].extend(data['Historical Context Deviation'].tolist())

    # Calculate the mean deviation for each window size
    mean_deviations = {
        window: sum(values) / len(values) for window, values in deviations_by_window.items()
    }

    # Sort the results by window size for consistent plotting
    sorted_windows = sorted(mean_deviations.keys())
    sorted_means = [mean_deviations[window] for window in sorted_windows]

    # Plot the mean deviation by window size
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(
        sorted_windows,
        sorted_means,
        marker='o',
        linestyle='-',
        label='Mean Deviation'  # Add legend label
    )
    plt.xlabel('Context Window Size')  # Label for x-axis
    plt.ylabel('Mean Historical Context Deviation')  # Label for y-axis
    plt.title('Mean Historical Context Deviation by Window Size')  # Title of the plot
    plt.grid(True)  # Enable grid for better readability
    plt.legend()  # Show legend on the plot

    # Save the plot to the specified folder
    plt.savefig(f"{output_folder_plots}/deviation_historical.analysis.mean.png")

    # Save the mean deviations to a CSV file
    df = pd.DataFrame({
        "Task Index": sorted_windows,
        "Historical Context Deviation": sorted_means
    })
    mean_output_file_path = os.path.join(output_folder_data,
                                         f"deviation_historical.analysis.mean.csv")
    df.to_csv(mean_output_file_path, index=False)

    # plt.show()  # Display the plot

    plt.clf()  # Clear the plot for future use


# Call the function to analyze and plot the data
analyze_and_plot_deviation_by_window(data_folder)