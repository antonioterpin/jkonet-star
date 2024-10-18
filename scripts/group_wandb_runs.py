import wandb
import re
from load_from_wandb import wandb_config 

# Authenticate with W&B (if necessary)
wandb.login()

# Initialize the API
api = wandb.Api()

# Get all runs from the project
runs = api.runs(f"{wandb_config["entity"]}/{wandb_config["project"]}")

split_pattern = re.compile(r"split_(\d+\.?\d*)")
dimension_pattern = re.compile(r"dim_(\d+)")
interaction_pattern = re.compile(r"interaction_(.*?)_dt")
sinkhorn_pattern = re.compile(r"sinkhorn_(\d+\.?\d*)")
for run in runs:
    GROUP_NAME = None
    if "RNA" in run.name:
        GROUP_NAME = "RNA"
    else:
        match = split_pattern.search(run.name)
        if not match:
            continue

        split_ratio = match.group(1)

        if "split_trajectories_True" in run.name:
            GROUP_NAME = "split-trajectories"
        else:
            GROUP_NAME = "split-particles"
        
        GROUP_NAME += f"-ratio_{split_ratio}"

        match = dimension_pattern.search(run.name)
        if match:
            dimension = match.group(1)
            if int(dimension) > 2:
                GROUP_NAME = "ablation-dimension"

        match = sinkhorn_pattern.search(run.name)
        if match:
            sinkhorn = match.group(1)
            if float(sinkhorn) > 1e-12:
                GROUP_NAME = "ablation-couplings"

        match = interaction_pattern.search(run.name)
        if match:
            interaction = match.group(1)
            if interaction != "none":
                GROUP_NAME = "all-energies"
    if GROUP_NAME is None:
        continue
    run.group = GROUP_NAME
    run.update()
    print(f"Updated run {run.name} with group: {GROUP_NAME}")