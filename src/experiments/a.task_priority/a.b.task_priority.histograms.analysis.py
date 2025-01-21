import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Path to the folder containing the CSV files (update this to your local folder)
folder_path = f"{current_dir}/experimental_data/data/priority_histograms"

# Initialize a list to store summary results for each file
summary = []

# Iterate over all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv') and not file_name.startswith('summary_results'):
        # Load the current CSV file
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # Calculate metrics for the current file
        avg_priority = df['priority'].mean()  # Average priority
        max_priority = df['priority'].max()  # Maximum priority
        critical_tasks = df[df['Critical'] == True]['priority'].mean()  # Average priority for critical tasks

        # Extract parameter values (Alpha, Beta, Gamma) from the file name
        alpha = file_name.split('_')[1].replace('alpha', '')
        beta = file_name.split('_')[2].replace('beta', '')
        gamma = file_name.split('_')[3].replace('gamma', '').replace('.csv', '')

        # Append the results to the summary list
        summary.append({
            'File': file_name,
            'Alpha': alpha,
            'Beta': beta,
            'Gamma': gamma,
            'Avg_Priority': avg_priority,
            'Max_Priority': max_priority,
            'Critical_Priority_Avg': critical_tasks
        })

# Convert the summary list into a DataFrame for analysis
summary_df = pd.DataFrame(summary)

# Define output folder for saving the summary CSV
output_folder_csv = f"{current_dir}/experimental_data/data"
os.makedirs(output_folder_csv, exist_ok=True)

# Save the summary as a CSV for local review (optional)
summary_df.to_csv(f"{output_folder_csv}/task_priority.analysis.summary.csv", index=False)

# Plot the relationship between Alpha and Avg_Priority, grouped by Gamma
plt.figure(figsize=(12, 6))
for gamma in summary_df['Gamma'].unique():
    # Filter data for the current Gamma value
    subset = summary_df[summary_df['Gamma'] == gamma]

    # Plot Alpha vs Avg_Priority for the current Gamma group
    plt.plot(subset['Alpha'], subset['Avg_Priority'], label=f'Gamma={gamma}')

# Add plot details
plt.title('Average Priority vs. Alpha (Grouped by Gamma)')
plt.xlabel('Alpha')
plt.ylabel('Average Priority')
plt.legend()
plt.grid()
# plt.show()

# Display summary results for manual review
print(summary_df)
