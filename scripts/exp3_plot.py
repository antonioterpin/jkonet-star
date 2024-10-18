import os
import wandb
import numpy as np
from tqdm import tqdm
from load_from_wandb import parse_name, wandb_config

os.makedirs('out/lightspeed', exist_ok=True)

api = wandb.Api()

group_name = 'all-energies'

runs = api.runs(f'{wandb_config['entity']}/{wandb_config['project']}', filters={"group": group_name})

per_method_data = {}
failed_runs = {}
for run in tqdm(runs):
    run_details = parse_name(run.name)
    potential = run_details['potential']
    interaction = run_details['interaction']
    beta = run_details['beta']
    method = run_details['method']

    if method not in per_method_data:
        per_method_data[method] = {
            'n_failed': 0,
            'errors': [],
            'exp': []
        }

    exp = f"{potential}|{interaction}|{beta}"
    if exp in per_method_data[method]['exp']:
        print(f'Duplicate: {run.name}')
    else:
        per_method_data[method]['exp'].append(exp)

    try:
        summary = run.summary
        if run.state != 'finished':
            if exp not in failed_runs:
                failed_runs[exp] = []
            failed_runs[exp].append(method)
            per_method_data[method]['n_failed'] += 1
            continue
        err = float(summary['error_w_one_ahead'])
        per_method_data[method]['errors'].append(err)
    except Exception as e:
        print(f'Error in {run.name}')
        print(e)

# Save config of failed runs
os.makedirs('out/all-energies', exist_ok=True)
with open('out/all-energies/failed_runs.sh', 'w') as f:
    for (exp, methods) in failed_runs.items():
        potential, interaction, beta = exp.split('|')
        if len(methods) > 1:
            f.write(f'python data_generator.py --potential {potential} --interaction {interaction} --n-particles 2000 --test-ratio 0.5 --internal wiener --beta {beta}\n')

with open('out/all-energies/failed_runs_unique.sh', 'w') as f:
    for (exp, methods) in failed_runs.items():
        if len(methods) == 1:
            potential, interaction, beta = exp.split('|')
            f.write('\n')
            f.write(f'python data_generator.py --potential {potential} --interaction {interaction} --n-particles 2000 --test-ratio 0.5 --internal wiener --beta {beta}\n')
            f.write(f'python train.py --dataset potential_{potential}_internal_wiener_beta_{beta}_interaction_{interaction}_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb --solver {methods[0]}\n')
        

methods = list(per_method_data.keys())

for method in methods:
    errors = per_method_data[method]['errors']
    n_failed = per_method_data[method]['n_failed']
    
    # Compute median, whiskers and quartiles
    errors = np.asarray(errors)
    median = np.median(errors)
    q1 = np.percentile(errors, 25)
    q3 = np.percentile(errors, 75)
    iqr = q3 - q1
    lower_whisker = np.max([np.min(errors), q1 - 1.5 * iqr])
    upper_whisker = np.min([np.max(errors), q3 + 1.5 * iqr])
    print(f'Method: {method}')
    print(f'N failed: {n_failed} out of {len(errors) + n_failed}: {n_failed / (len(errors) + n_failed) * 100:.2f}%')
    print(f'Median: {median}')
    print(f'Q1: {q1}')
    print(f'Q3: {q3}')
    print(f'IQR: {iqr}')
    print(f'Lower whisker: {lower_whisker}')
    print(f'Upper whisker: {upper_whisker}')
    print("-----------------------------")