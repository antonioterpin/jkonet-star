import os
import wandb
import numpy as np
from tqdm import tqdm
from load_from_wandb import parse_name, wandb_config

folder = 'out/ablation-couplings'
os.makedirs(folder, exist_ok=True)

api = wandb.Api()

group_name = 'ablation-couplings'

runs = api.runs(f'{wandb_config['entity']}/{wandb_config['project']}', filters={"group": group_name})
per_epsilon_data = {}

def add_run_data(run, eps, potential):
    summary = run.summary
    err = float(summary['error_w_one_ahead'])
    if run.state != 'finished':
        err = np.nan
    per_epsilon_data[eps][potential] = err

max_error = -np.inf
for run in tqdm(runs):
    run_details = parse_name(run.name)
    potential = run_details['potential']
    eps = run_details['sinkhorn']

    if eps not in per_epsilon_data:
        per_epsilon_data[eps] = {}

    try:
        add_run_data(run, eps, potential)
    except Exception as e:
        print(f'Error in {run.name}')
        print(e)

sinkhorn_eps = list(per_epsilon_data.keys())

if '0.0' not in sinkhorn_eps:
    per_epsilon_data['0.0'] = {}
    # Pick 0.0 from lightspeed group
    runs = api.runs(f'{wandb_config['entity']}/{wandb_config['project']}', filters={"group": 'split-trajectories-ratio_0.5'})
    for run in tqdm(runs):
        run_details = parse_name(run.name)
        potential = run_details['potential']
        method = run_details['method']
        if method == 'jkonet-star-potential':
            add_run_data(run, '0.0', potential)

sinkhorn_eps += ['0.0']

# In the order of the paper
potentials = [
    'styblinski_tang',
    'holder_table',
    'flowers',
    'oakley_ohagan',
    'watershed',
    'ishigami',
    'friedman',
    'sphere',
    'bohachevsky',
    'wavy_plateau',
    'zigzag_ridge',
    'double_exp',
    'relu',
    'rotational',
    'flat'
]

# Save normalized errors
max_val = np.nanmax([per_epsilon_data[eps][potential] for eps in sinkhorn_eps for potential in potentials])
min_val = np.nanmin([per_epsilon_data[eps][potential] for eps in sinkhorn_eps for potential in potentials])
with open(f'{folder}/error.csv', 'w') as file:
    for eps in sinkhorn_eps:
        errors = np.asarray([per_epsilon_data[eps][potential] for potential in potentials])
        errors = (errors - min_val) / (max_val - min_val)
        file.write(f'{eps} {np.mean(errors)} {np.std(errors)}\n')

# If times are saved from the terminal (TODO: log them in a more convenient way)
if os.path.exists(f'{folder}/raw_times/'):
    with open(f'{folder}/times.txt', 'a') as file_times:
        for eps in sinkhorn_eps:
            times = []
            for potential in potentials:
                with open(f'{folder}/raw_times/{potential}_{eps}.txt') as file:
                    for line in file:
                        times.append(float(line))
        
            file_times.write(f'{eps} {np.mean(times)} {np.std(times)}\n')