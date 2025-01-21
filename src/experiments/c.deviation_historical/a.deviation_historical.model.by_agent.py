import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

from src.utils.files import load_from_json

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Define the folder containing experiment data
experiment_data_folder = f"{current_dir}/../.data"

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

# Fibonacci sequence for defining context windows
fibonacci = [1, 2, 3, 5, 8, 13, 21]

# Load deviation data for agents
agent_deviations = load_from_json(f"{experiment_data_folder}/agent_deviations.json")

# Define output folders for data and plots
output_folder_data = f"{current_dir}/experimental_data/data/model/deviation_historical"
os.makedirs(output_folder_data, exist_ok=True)
output_folder = f"{current_dir}/experimental_data/plots/model/deviation_historical"
os.makedirs(output_folder, exist_ok=True)

# Plot deviations for each agent
plt.figure(figsize=(12, 6))
for agent, agent_list_deviation in agent_deviations.items():
    for experiment_window_size, agent_deviation in agent_list_deviation.items():
        plt.plot(range(1, len(agent_deviation) + 1), agent_deviation,
                 label=f"Context Window - {experiment_window_size}")
        agent_deviation_data = list(zip(range(1, len(agent_deviation) + 1), agent_deviation))

        # Save deviation data for each agent and context window
        df = pd.DataFrame(agent_deviation_data, columns=["Task Index", "Historical Context Deviation"])
        output_file_path = os.path.join(output_folder_data,
                                        f"summary_results.agent.{agent_model_names[agent]}.window_{experiment_window_size}.csv")
        df.to_csv(output_file_path, index=False)

    # Plot details
    plt.xlabel("Task Index")
    plt.ylabel("Deviation (1 - Cosine Similarity)")
    plt.title(f"Deviation for Agent \"{agent_model_names[agent]}\".")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot for the current agent
    output_file = os.path.join(output_folder,
                               f'deviations_analysis_chart.agent.{agent_model_names[agent]}.png')
    plt.savefig(output_file)

    # Clear the plot for the next agent
    plt.clf()

# Analyze deviations for each context window
for context_window in fibonacci:
    agent_stats = []  # Data for each agent (average, max, min deviations)
    agent_names = []  # List of agent names

    for agent, context_data in agent_deviations.items():
        if str(context_window) in context_data:
            deviations = context_data[str(context_window)]
            avg_deviation = np.mean(deviations)
            max_deviation = np.max(deviations)
            min_deviation = np.min(deviations)

            agent_stats.append([avg_deviation, max_deviation, min_deviation])
            agent_names.append(agent)

    if agent_stats:  # Only plot if data exists
        agent_stats = np.array(agent_stats)

        # Create bar plot for deviations
        x = np.arange(len(agent_names))  # Indices for each agent
        bar_width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - bar_width, agent_stats[:, 0], width=bar_width, label="Average Deviation", color="blue")
        plt.bar(x, agent_stats[:, 1], width=bar_width, label="Maximum Deviation", color="green")
        plt.bar(x + bar_width, agent_stats[:, 2], width=bar_width, label="Minimum Deviation", color="red")

        # Plot details
        plt.title(f"Deviations for Context Window {context_window}")
        plt.xlabel("Agents")
        plt.ylabel("Deviation (1 - Cosine Similarity)")
        plt.xticks(x, agent_names, rotation=45)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Add description of agent models
        description = "\n".join([f"{short}: {full}" for short, full in agent_model_names.items()])
        plt.figtext(0.99, 0.01, description, horizontalalignment="right", fontsize=10, wrap=True)

        # Save the plot
        plt.savefig(f"{output_folder}/deviation_historical.agents.{context_window}.png")

        # Clear the plot for the next context window
        plt.clf()