import os
import wandb
import numpy as np
from tqdm import tqdm
from load_from_wandb import parse_name, wandb_config

folder = 'out/ablation-dimension'

# Create directory to store the CSV files
os.makedirs(folder, exist_ok=True)

# Initialize the Weights and Biases API
api = wandb.Api()

# Define the group name
group_name = 'ablation-dimension'

# Fetch the runs that match the group name
runs = api.runs(f"{wandb_config['entity']}/{wandb_config['project']}", filters={"group": group_name})

# Prepare a dictionary to store the data per potential
per_potential_data = {}

# Loop through each run
for run in tqdm(runs):
    # Parse the run name to extract details
    run_details = parse_name(run.name)
    potential = run_details['potential']
    dimension = run_details['dim']
    n_particles = run_details['N']

    # Extract the error from the run summary
    error = run.summary.get('error_w_one_ahead', None)

    # Skip if error is not present
    if error is None:
        print(f"Error not found in run: {run.name}")
        continue

    # Initialize nested dictionary if potential is not already in it
    if potential not in per_potential_data:
        per_potential_data[potential] = []

    # Append the data in the format (dimension, n_particles, error)
    per_potential_data[potential].append((int(dimension), int(n_particles), float(error)))

# For each potential, create a CSV file with the data
for potential, data in per_potential_data.items():
    # Sort data by dimension and n_particles for easier reading
    data.sort(key=lambda x: (x[1], x[0]))

    # Define the output file path
    output_file = f'{folder}/{potential}.csv'

    # Write the data to the CSV file
    with open(output_file, mode='w', newline='') as csv_file:
        # Write the header
        csv_file.write('x y z\n')

        # Initialize variables to track the current n_particles
        current_n_particles = None

        # Write the data with fixed precision format
        for dimension, n_particles, error in data:
            # If the n_particles changes, add a blank line between groups
            if current_n_particles is not None and current_n_particles != n_particles:
                csv_file.write('\n')

            # Write the data in the required format with fixed precision
            csv_file.write(f"{int(dimension)} {int(n_particles/2)} {error:.10f}\n")

            # Update the current n_particles
            current_n_particles = n_particles