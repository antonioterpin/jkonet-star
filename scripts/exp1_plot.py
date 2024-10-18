import os
import wandb
import numpy as np
from tqdm import tqdm
from load_from_wandb import parse_name, wandb_config

folder = 'out/lightspeed'
os.makedirs(folder, exist_ok=True)

api = wandb.Api()

group_name = 'split-trajectories-ratio_0.5'

runs = api.runs(f'{wandb_config['entity']}/{wandb_config['project']}', filters={"group": group_name})

per_method_data = {}
max_error = -np.inf
for run in tqdm(runs):
    run_details = parse_name(run.name)
    potential = run_details['potential']
    method = run_details['method']

    if method not in per_method_data:
        per_method_data[method] = {}


    try:
        summary = run.summary
        err = float(summary['error_w_one_ahead'])
        err_std = float(summary['error_w_one_ahead_std'])
        if run.state != 'finished':
            err = np.nan
            err_std = np.nan
        per_method_data[method][potential] = {
            'error_avg': err,
            'error_std': err_std,
        }
        max_error = np.nanmax([max_error, err])

        time = []
        errors = []
        for entry in run.scan_history():
            if entry['time']:
                time.append(entry['time'])
            if entry['error_w_one_ahead']:
                errors.append(float(entry['error_w_one_ahead']))
        per_method_data[method][potential]['time_avg'] = np.average(time)
        per_method_data[method][potential]['time_std'] = np.std(time)
        per_method_data[method][potential]['all_errors_avg'] = np.asarray(errors)
    except Exception as e:
        print(f'Error in {run.name}')
        print(e)

methods = list(per_method_data.keys())

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
with open(f'{folder}/error.csv', 'w') as file:
    file.write(','.join(['exp'] + [method for method in methods]) + '\n')
    for (i, potential) in enumerate(potentials):
        file.write(','.join([str(i+1)] + [
            str(10 if np.isnan(per_method_data[method][potential]['error_avg']) else per_method_data[method][potential]['error_avg']) 
            for method in methods]) + '\n')

# Save times
with open(f'{folder}/time.csv', 'w') as file:
    file.write("method,median,q1,q3,lw,uw\n")
    for method in methods:
        data = np.asarray([per_method_data[method][potential]['time_avg'] for potential in potentials])
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lw = np.min(data[data >= q1 - 1.5 * iqr])
        uw = np.max(data[data <= q3 + 1.5 * iqr])
        file.write(f"{method},{median},{q1},{q3},{lw},{uw}\n")

# Save learning curves
n_max_samples = np.max([len(per_method_data[method][potential]['all_errors_avg']) for method in methods for potential in potentials])
for method in methods:
    error_trajectories = []
    for potential in potentials:
        error_trajectory = per_method_data[method][potential]['all_errors_avg']
        error_trajectory = (error_trajectory - np.nanmin(error_trajectory)) / (np.nanmax(error_trajectory) - np.nanmin(error_trajectory))
        error_trajectory = np.pad(error_trajectory, (0, n_max_samples - len(error_trajectory)), constant_values=10)
        if len(error_trajectory) != n_max_samples:
            print(f"Error in {method} {potential}")
        if len(error_trajectories) == 0:
            error_trajectories = error_trajectory
        else:
            error_trajectories = np.vstack([error_trajectories, error_trajectory])
    
    learning_curve = np.nanmean(error_trajectories, axis=0)
    learning_curve_std = np.nanstd(error_trajectories, axis=0)

    with open(f'{folder}/error_wasserstein_{method}.txt', 'w') as file:
        file.write(f'1 0\n')
        for (err, err_std) in zip(learning_curve, learning_curve_std):
            file.write(f'{err} {err_std}\n')