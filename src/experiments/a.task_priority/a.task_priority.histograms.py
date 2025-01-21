import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Load the data from a CSV file
data = pd.read_csv(f'{current_dir}/../.data/priority_tasks.generated.csv')

# Function to calculate task priorities based on weighting coefficients
def calculate_task_priority(data, alpha, beta, gamma):
    """
    Calculate the task priority considering three factors:
    - Urgency (weighted by alpha)
    - Resource availability (weighted by beta)
    - System load (weighted by gamma)

    Parameters:
    - data: DataFrame containing task attributes.
    - alpha: Weight coefficient for urgency.
    - beta: Weight coefficient for resource availability (inverse relation).
    - gamma: Weight coefficient for system load.

    Returns:
    - Array of calculated task priorities.
    """
    priorities = []
    for _, row in data.iterrows():
        # Safeguard against zero or missing values to avoid division by zero
        urgency = row['Urgency'] if row['Urgency'] > 0 else 1e-6
        resources = row['Resources'] if row['Resources'] > 0 else 1e-6
        system_load = row['System_Load'] if row['System_Load'] > 0 else 1e-6

        # Priority calculation formula
        priority = alpha * urgency + beta * (1 / resources) + gamma * system_load

        priorities.append(priority)

    return np.array(priorities)

# Define ranges of weighting factors for sensitivity analysis
alpha_values = [0.01, 0.2, 0.5, 0.9, 1]
beta_values = [0.01, 0.2, 0.5, 0.9, 1]
gamma_values = [0.01, 0.2, 0.5, 0.9, 1]

# Placeholder for storing results and matrices for analysis
results = []
mat = []

# Define output folders for saving plots and CSV data
output_folder = f"{current_dir}/experimental_data/plots/priority_histograms"
os.makedirs(output_folder, exist_ok=True)

output_folder_csv = f"{current_dir}/experimental_data/data/priority_histograms"
os.makedirs(output_folder_csv, exist_ok=True)

# Iterate over all combinations of weighting factors
for alpha in alpha_values:
    for beta in beta_values:
        for gamma in gamma_values:
            # Calculate priorities for the current combination of weights
            data['priority'] = calculate_task_priority(data, alpha=alpha, beta=beta, gamma=gamma)

            # Create a histogram of task priorities
            plt.figure(figsize=(10, 6))
            plt.hist(data['priority'], bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'Histogram of Task Priorities (alpha={alpha}, beta={beta}, gamma={gamma})')
            plt.xlabel('Task Priority')
            plt.ylabel('Frequency')
            plt.grid(True)

            # Save the calculated priorities to a CSV file
            output_file_csv = os.path.join(output_folder_csv, f'histogram_alpha{alpha}_beta{beta}_gamma{gamma}.csv')
            data.to_csv(output_file_csv, index=False)

            # Save the histogram plot as a PNG file
            output_file = os.path.join(output_folder, f'histogram_alpha{alpha}_beta{beta}_gamma{gamma}.png')
            plt.savefig(output_file)
            plt.close()

# Indicate that the analysis and plotting are complete
print("Tests completed. Priority histograms with different weights plotted.")
print(mat)
