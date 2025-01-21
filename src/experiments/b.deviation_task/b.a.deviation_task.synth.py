import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Experiment parameters
num_tasks = 50  # Number of tasks
np.random.seed(42)  # For reproducibility

def lambda_sin(n_sin, amplitude=5, frequency=1, phase_shift=0, offset=0):
    """
    Generate sinusoidal data for a given number of points.

    Parameters:
    - n_sin: Number of points.
    - amplitude: Maximum amplitude of the sinusoid.
    - frequency: Frequency of the sinusoid.
    - phase_shift: Phase shift of the sinusoid.
    - offset: Vertical offset of the sinusoid.

    Returns:
    - x: Array of x values.
    - y: Array of y values based on the sinusoidal function.
    """
    x = np.linspace(0, 2 * np.pi, n_sin)  # N points over the interval [0, 2*pi]
    y = (amplitude / 2) * (1 + np.sin(frequency * x + phase_shift)) + offset
    return x, y

def optimal_cos(n_cos, amplitude=5, frequency=1, phase_shift=0, offset=0):
    """
    Generate cosinusoidal data for a given number of points.

    Parameters:
    - n_cos: Number of points.
    - amplitude: Maximum amplitude of the cosine.
    - frequency: Frequency of the cosine.
    - phase_shift: Phase shift of the cosine.
    - offset: Vertical offset of the cosine.

    Returns:
    - x: Array of x values.
    - y: Array of y values based on the cosinusoidal function.
    """
    x = np.linspace(0, 2 * np.pi, n_cos)  # N points over the interval [0, 2*pi]
    y = (amplitude / 2) * (1 + np.cos(frequency * x + phase_shift)) + offset
    return x, y

# Generate data for tasks
tasks = {
    "task_id": np.arange(1, num_tasks + 1),
    "len_optimal": np.random.randint(20, 200, num_tasks),
    "len_output": np.random.randint(40, 160, num_tasks),
}

# Create a DataFrame
df = pd.DataFrame(tasks)

# Calculate deviation between optimal and output lengths
df["deviation"] = df["len_optimal"] - df["len_output"]

# Aggregated metrics
average_deviation = df["deviation"].mean()
max_deviation = df["deviation"].max()
min_deviation = df["deviation"].min()

# Display results
print("Task-specific Deviation Analysis")
print(df)
print(f"\nAverage Deviation: {average_deviation}")
print(f"Max Deviation: {max_deviation}")
print(f"Min Deviation: {min_deviation}")

# Define output folders for plots and data
output_folder = f"{current_dir}/experimental_data/plots/synth"
os.makedirs(output_folder, exist_ok=True)

# Visualization
plt.figure(figsize=(12, 6))
plt.bar(df["task_id"], df["deviation"], color='blue', alpha=0.7, label="Deviation")
plt.plot(df["task_id"], df["len_optimal"], color='green', label="Optimal Length")
plt.scatter(df["task_id"], df["len_output"], color='red', label="Response Length")
plt.xlabel("Task ID")
plt.ylabel("Length/Deviation")
plt.title("Task-specific Deviation Analysis")
plt.legend()
plt.grid()

# Save the plot as a PNG file
output_file = os.path.join(output_folder, f'deviation_task.synth.png')
plt.savefig(output_file)

# Define output folder for data
output_folder_data = f"{current_dir}/experimental_data/data/synth"
os.makedirs(output_folder_data, exist_ok=True)

# Save the DataFrame to a CSV file
df.to_csv(f"{output_folder_data}/deviation_task.synth.csv", index=False)

# plt.show()