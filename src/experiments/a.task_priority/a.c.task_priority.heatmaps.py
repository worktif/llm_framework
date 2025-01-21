import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Load data from the specified CSV file
file_path = f"{current_dir}/experimental_data/data/task_priority.analysis.summary.csv"  # Specify the path to your CSV file
data = pd.read_csv(file_path)

# Define output folder for saving heatmaps
output_folder_plots = f"{current_dir}/experimental_data/plots/heatmaps"
os.makedirs(output_folder_plots, exist_ok=True)

# Unique values for Alpha, Beta, and Gamma
alpha_values = [0.01, 0.2, 0.5, 0.9, 1]
beta_values = [0.01, 0.2, 0.5, 0.9, 1]
gamma_values = [0.01, 0.2, 0.5, 0.9, 1]

# Generate heatmaps for all combinations
for alpha in alpha_values:
    # Filter data for the current Alpha value
    filtered_data = data[data['Alpha'] == alpha]
    heatmap_data = filtered_data.groupby(['Beta', 'Gamma'])['Max_Priority'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot_table(index='Beta', columns='Gamma', values='Max_Priority')

    # Fill missing values to avoid empty areas in the heatmap
    heatmap_pivot = heatmap_pivot.reindex(index=beta_values, columns=gamma_values, fill_value=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_pivot, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Max Priority'})
    plt.title(f'Heatmap of Max Priority (Alpha={alpha})')
    plt.xlabel('Gamma')
    plt.ylabel('Beta')

    plt.savefig(f"{output_folder_plots}/task_priority.heatmap.lambda_differences.alpha-{alpha}.png")

    # Clear the figure to avoid overlaps in subsequent plots
    plt.clf()

for beta in beta_values:
    # Filter data for the current Beta value
    filtered_data = data[data['Beta'] == beta]
    heatmap_data = filtered_data.groupby(['Alpha', 'Gamma'])['Max_Priority'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot_table(index='Alpha', columns='Gamma', values='Max_Priority')

    # Fill missing values
    heatmap_pivot = heatmap_pivot.reindex(index=alpha_values, columns=gamma_values, fill_value=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_pivot, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Max Priority'})
    plt.title(f'Heatmap of Max Priority (Beta={beta})')
    plt.xlabel('Gamma')
    plt.ylabel('Alpha')

    plt.savefig(f"{output_folder_plots}/task_priority.heatmap.lambda_differences.beta-{beta}.png")

    # Clear the figure to avoid overlaps in subsequent plots
    plt.clf()

for gamma in gamma_values:
    # Filter data for the current Gamma value
    filtered_data = data[data['Gamma'] == gamma]
    heatmap_data = filtered_data.groupby(['Alpha', 'Beta'])['Max_Priority'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot_table(index='Alpha', columns='Beta', values='Max_Priority')

    # Fill missing values
    heatmap_pivot = heatmap_pivot.reindex(index=alpha_values, columns=beta_values, fill_value=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_pivot, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Max Priority'})
    plt.title(f'Heatmap of Max Priority (Gamma={gamma})')
    plt.xlabel('Beta')
    plt.ylabel('Alpha')

    plt.savefig(f"{output_folder_plots}/task_priority.heatmap.lambda_differences.gamma-{gamma}.png")

    # Clear the figure to avoid overlaps in subsequent plots
    plt.clf()
