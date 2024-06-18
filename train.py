import os
import jax
import yaml
import torch
import wandb
import argparse
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import EnumMethod, get_model
from dataset import PopulationEvalDataset
from utils.sde_simulator import SDESimulator
from utils.plotting import plot_level_curves, plot_predictions

# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def main(args):
    # Set random key
    key = jax.random.PRNGKey(args.seed)

    # Load config
    config = yaml.safe_load(open('config.yaml'))
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    save_locally = config['train']['save_locally']

    # Load also JKOnet configs, although they may not be used
    jkonet_config = yaml.safe_load(open('config-jkonet-extra.yaml'))
    # merge configs
    config.update(jkonet_config)

    # Initialize wandb
    if args.wandb:
        wandb.init(
            project='jkonet-star',
            config=config)
        wandb.run.name = f"{args.solver}.{args.dataset}.{args.seed}"

    # Load model and dataset
    dataset_eval = PopulationEvalDataset(key, args.dataset)
    model = get_model(
        args.solver, config, 
        dataset_eval.data_dim, dataset_eval.dt)
    state = model.create_state(key)
    dataset_train = model.load_dataset(args.dataset)
    torch.manual_seed(args.seed)
    loader_train = DataLoader(
            dataset_train, batch_size=batch_size if batch_size > 0 else len(dataset_train), shuffle=True, collate_fn=numpy_collate)
    loader_val = DataLoader(
            dataset_eval, batch_size=len(dataset_eval), shuffle=False,  collate_fn=numpy_collate)

    # Train
    epochs = config['train']['epochs']
    eval_freq = config['train']['eval_freq']
    progress_bar = tqdm(range(1, epochs + 1))
    train_step = model.train_step
    if epochs > 1:
        train_step = jax.jit(model.train_step)
    for epoch in progress_bar:
        loss = 0
        t_start = time()
        for sample in loader_train:
            l, state = train_step(state, sample)
            loss += l

        t_end = time()
        
        progress_bar.desc = f"Epoch {epoch} | Loss: {loss / len(loader_train)}"
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'time': t_end - t_start,
                'loss': loss.item() / len(loader_train)})
        
        if epoch % eval_freq == 0:
            key, key_eval = jax.random.split(key)
            init_pp = next(iter(loader_val))
            potential = model.get_potential(state)
            beta = model.get_beta(state)
            interaction = model.get_interaction(state)
            predictions = SDESimulator(
                    dataset_eval.dt,
                    dataset_eval.T,
                    potential,
                    beta,
                    interaction
                ).forward_sampling(key_eval, init_pp)
            
            # plot trajectories
            plot_folder_name = os.path.join('out', 'plots', args.dataset, str(args.solver), str(epoch))
            if save_locally:
                # saving plots to file
                # create folder if not exists
                if not os.path.exists(plot_folder_name):
                    os.makedirs(plot_folder_name)
            if args.wandb and config['wandb']['save_model']:
                potential_params, internal_params, interaction_params = model.get_params(state)
                wandb.log({
                    'potential_parameters': potential_params,
                    'internal_parameters': internal_params,
                    'interaction_parameters': interaction_params,
                })

            if save_locally or args.wandb:
                plot_path = os.path.join(plot_folder_name, 'predictions') if save_locally else None
                trajectory_fig = plot_predictions(predictions, dataset_eval.trajectory, model=str(args.solver), save_to=plot_path)

                level_curves_potential_fig = plot_level_curves(
                    potential, ((-4, -4), (4, 4)), dimensions=dataset_eval.data_dim,
                    save_to=os.path.join(plot_folder_name, 'level_curves_potential') if save_locally else None
                )

                level_curves_interaction_fig = plot_level_curves(
                    interaction, ((-4, -4), (4, 4)), dimensions=dataset_eval.data_dim,
                    save_to=os.path.join(plot_folder_name, 'level_curves_interaction') if save_locally else None
                )

                if args.wandb and config['wandb']['save_plots']:
                    wandb.log({
                        'epoch': epoch,
                        'trajectory': wandb.Image(trajectory_fig),
                        'level_curves_potential': wandb.Image(level_curves_potential_fig),
                        'level_curves_interaction': wandb.Image(level_curves_interaction_fig)
                    })
                # close figs
                plt.close(trajectory_fig)
                plt.close(level_curves_potential_fig)
                plt.close(level_curves_interaction_fig)

            # compute errors
            error_wasserstein = dataset_eval.error_wasserstein(predictions)

            if dataset_eval.no_ground_truth:
                error_potential = 0
                error_internal = 0
                error_interaction = 0
            else:
                error_potential = dataset_eval.error_potential(
                    SDESimulator(
                        dataset_eval.dt,
                        dataset_eval.T,
                        potential,
                        False,
                        False
                    ).forward_sampling(key_eval, init_pp)
                )
                error_internal = dataset_eval.error_internal(beta)
                error_interaction = dataset_eval.error_interaction(
                    SDESimulator(
                        dataset_eval.dt,
                        dataset_eval.T,
                        False,
                        False,
                        interaction
                    ).forward_sampling(key_eval, init_pp)
                )

            print(f"Epoch {epoch} | Wasserstein: {error_wasserstein} | Potential: {error_potential} | Internal: {error_internal} | Interaction: {error_interaction}")

            if args.wandb:
                wandb.log({
                    'epoch': epoch,
                    'error_wasserstein': float(error_wasserstein),
                    'error_potential': float(error_potential),
                    'error_internal': float(error_internal),
                    'error': float(beta),
                    'error_interaction': float(error_interaction)
                })

            # # Save model
            # model.save(f"models/{args.solver}_{args.dataset}_{args.seed}_{epoch}.pt")

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--solver', '-s',
        type=EnumMethod, 
        choices=list(EnumMethod), 
        default=EnumMethod.JKO_NET_STAR_POTENTIAL,
        help=f"""Name of the solver to use.""",
        )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str, 
        help=f"""Dataset to train the model on.""",
        )
    
    parser.add_argument('--wandb', action='store_true',
                        help='Option to run with activated wandb.')

    parser.add_argument('--debug', action='store_true',
                        help='Option to run in debug mode.')
    
    # reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Set seed for the run'
    )

    args = parser.parse_args()

    # set debug mode
    if args.debug:
        print('Running in DEBUG mode.')
        jax.config.update('jax_disable_jit', True)

    main(args)