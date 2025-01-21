import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Define output folders for plots and data
output_folder_plots = f"{current_dir}/experimental_data/plots/model/llm_formula"
os.makedirs(output_folder_plots, exist_ok=True)

output_folder_data = f"{current_dir}/experimental_data/data/model/llm_formula"
os.makedirs(output_folder_data, exist_ok=True)

# Path to the task-specific deviation data
deviation_task_file = f"{current_dir}/../b.deviation_task/experimental_data/data/model/deviation_task/deviation_task.analysis.summary.csv"  # Replace with your file path

# Mapping of agent names to model identifiers
agent_model_names = {
    "A1": 'all-MiniLM-L6-v2',
    "A2": 'paraphrase-MiniLM-L6-v2',
    "A3": 'paraphrase-albert-small-v2',
    "A4": 'all-MiniLM-L12-v2',
    "A5": 'paraphrase-multilingual-MiniLM-L12-v2',
}

# Initialize agent models
agent_models = {
    "A1": SentenceTransformer('all-MiniLM-L6-v2'),
    "A2": SentenceTransformer('paraphrase-MiniLM-L6-v2'),
    "A3": SentenceTransformer('paraphrase-albert-small-v2'),
    "A4": SentenceTransformer('all-MiniLM-L12-v2'),
    "A5": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),
}

# Fibonacci sequence for context window sizes
fibonacci = [1, 2, 3, 5, 8, 13, 21]

# Set the number of rows and columns for subplot layout
plot_rows = 42
plot_columns = 3

# Initialize a large figure for subplotting
fig, axes = plt.subplots(plot_rows, plot_columns, figsize=(50, 100))
axes_list = axes.flat

# Load the task-specific deviation data
data_task_specific = pd.read_csv(deviation_task_file)

iteration = 0

# Set rolling window size for calculating minimum L values
l_window = 5

for agent, agent_name in agent_model_names.items():
    for i, experiment_window_size in enumerate(fibonacci):
        # Path to historical deviation data for the current agent and window size
        file_path = f"{current_dir}/../c.deviation_historical/experimental_data/data/model/deviation_historical/summary_results.agent.{agent_name}.window_{experiment_window_size}.csv"
        df = pd.read_csv(file_path).sort_values(by="Task Index")

        # Merge datasets on the "Task Index" column
        data = pd.merge(
            data_task_specific,
            df,
            left_on='task_id',
            right_on='Task Index',
            how='inner'
        )

        # Calculate the context deviation and L values
        data['context_deviation'] = data['deviation'] * df['Historical Context Deviation']  # \(\mathcal{D}_{\text{context}}\)
        data['L'] = data['len_output'] + data['context_deviation']  # Final L value

        # Calculate the rolling minimum of L values
        data['L_min_window'] = data['L'].rolling(window=l_window, min_periods=1).min()

        # Create the plot for L values and components
        plt.figure(figsize=(12, 6))
        plt.plot(data['task_id'], data['L'], label=f'L - Objective Function Value. Agent: "{agent_name}". Context Window:  {experiment_window_size}', marker='o', color='darkgreen')
        plt.plot(data['task_id'], data['len_output'], label='Response Length', linestyle='--', color='orange')
        plt.plot(data['task_id'], data['context_deviation'], label='Context Deviation', linestyle=':', color='teal')
        plt.plot(data['task_id'], data['len_optimal'], label='Optimal Response', linestyle='-.', color='purple')

        # Add labels, title, and legend
        plt.xlabel('Task ID')
        plt.ylabel('Metric Value')
        plt.title(f'L - Function Analysis: Agent: "{agent_name}", Context Window:  {experiment_window_size}')
        plt.legend()
        plt.grid(True)

        # Save the figure to the output folder
        output_file = os.path.join(output_folder_plots,
                                   f'l_function_analysis.{agent_name}.window_{experiment_window_size}.png')
        plt.savefig(output_file)

        # Save the final dataset with computed values to a CSV file
        data.to_csv(f"{output_folder_data}/summary_results_with_L.{agent_name}.window_{experiment_window_size}.csv",
                    index=False)

        # Clear the plot to avoid overlap
        plt.clf()
