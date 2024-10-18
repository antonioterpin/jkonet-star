"""
This module provides a script for training and evaluating JKOnet* and other models for learning diffusion terms on population data.

Functions
-------------
- ``numpy_collate``
    A custom collate function for PyTorch's DataLoader to properly stack or nest NumPy arrays when using JAX.

- ``main``
    The main function that orchestrates the training loop, evaluation, logging, and visualization. It reads configurations, initializes models and datasets, and executes the training and evaluation processes.

Command-Line arguments
----------------------
The script accepts the following command-line arguments:

- `--solver`, `-s` (`EnumMethod`):
    Name of the solver (model) to use. Choices are defined in the `EnumMethod` class.

- `--dataset`, `-d` (`str`):
    Name of the dataset to train the model on. The dataset should be prepared and located in a directory matching this name.

- `--eval` (`str`):
    Option to test the fit on `'train_data'` or `'test_data'` (e.g., for debugging purposes). Default is `'test_data'`.

- `--wandb` (`bool`):
    If specified, activates Weights & Biases logging for experiment tracking.

- `--debug` (`bool`):
    If specified, runs the script in debug mode (disables JIT compilation in JAX for easier debugging).

- `--seed` (`int`):
    Seed for random number generation to ensure reproducibility.

- `--epochs` (`int`):
    Number of epochs to train the model. If not specified, the number of epochs is taken from the configuration file.

Usage example
-------------
To train a model using the `jkonet-star-potential` solver on a dataset named `my_dataset` with wandb logging:

.. code-block:: bash

    python train.py --solver jkonet-star-potential --dataset my_dataset --wandb

"""

import os
import jax
import yaml
import torch
import wandb
import argparse
import chex
from time import time
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import EnumMethod, get_model
from dataset import PopulationEvalDataset
from utils.sde_simulator import get_SDE_predictions
from utils.plotting import plot_level_curves, plot_predictions
from typing import Union, List, Tuple
from scripts.load_from_wandb import wandb_config

def numpy_collate(batch: List[Union[np.ndarray, Tuple, List]]) -> Union[np.ndarray, List]:
    """
    Collates a batch of samples into a single array or nested list of arrays.

    This function recursively processes a batch of samples, stacking NumPy arrays, and collating lists or tuples by grouping elements together. If the batch consists of NumPy arrays, they are stacked. If the batch contains tuples or lists, the function recursively applies the collation.

    This collate function is taken from the `JAX tutorial with PyTorch Data Loading <https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html>`_.

    Parameters
    ----------
    batch : List[Union[np.ndarray, Tuple, List]]
        A batch of samples where each sample is either a NumPy array, a tuple, or a list. It depends on the
        data loader.

    Returns
    -------
    np.ndarray
        The collated batch, either as a stacked NumPy array or as a nested structure of arrays.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def main(args: argparse.Namespace) -> None:
    """
    Main function to train a model on a specified dataset and evaluate it.

    Parameters
    ----------
    args : argparse.Namespace (see module description)

    Returns
    -------
    None
        The function trains the model, evaluates it, and optionally logs the results and metrics.
    """
    # Set random key
    key = jax.random.PRNGKey(args.seed)

    # Load config
    config = yaml.safe_load(open('config.yaml'))
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    eval_freq = config['train']['eval_freq']
    save_locally = config['train']['save_locally']

    chex.assert_scalar_positive(epochs)
    chex.assert_scalar_positive(batch_size)
    chex.assert_scalar_positive(eval_freq)

    # Load also JKOnet configs, although they may not be used
    jkonet_config = yaml.safe_load(open('config-jkonet-extra.yaml'))
    # merge configs
    config.update(jkonet_config)

    if args.epochs:
        # override epochs
        config['train']['epochs'] = args.epochs


    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=wandb_config['project'],
            config=config)
        wandb.run.name = f"{args.solver}.{args.dataset}.{args.seed}"

    # Load model and dataset
    dataset_eval = PopulationEvalDataset(
        key,
        args.dataset,
        str(args.solver),
        config['metrics']['wasserstein_error'],
        args.eval,
        )
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

    print(f"Training {args.solver} on {args.dataset} with seed {args.seed} for {config['train']['epochs']} epochs.")

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
            predictions = get_SDE_predictions(
                    str(args.solver),
                    dataset_eval.dt,
                    dataset_eval.T,
                1,
                    potential,
                    beta,
                    interaction,
                    key_eval,
                    init_pp)

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

            if save_locally or (args.wandb and config['wandb']['save_plots']):
                plot_path = os.path.join(plot_folder_name, 'predictions') if save_locally else None
                trajectory_fig = plot_predictions(
                    predictions,
                    dataset_eval.trajectory,
                    interval=None,
                    model=str(args.solver),
                    save_to=plot_path)

                if str(args.solver) != 'jkonet-star-time-potential':
                    level_curves_potential_fig = plot_level_curves(
                        potential, ((-4, -4), (4, 4)), dimensions=dataset_eval.data_dim,
                        save_to=os.path.join(plot_folder_name, 'level_curves_potential') if save_locally else None
                    )
                else:
                    for t in range(1, dataset_eval.T+1):
                        # Generate the potential function for the current time
                        current_potential = lambda x: potential(jnp.concatenate([x, jnp.array([t])], axis=0))
                        level_curves_potential_fig = plot_level_curves(
                            current_potential, ((-4, -4), (4, 4)), dimensions=dataset_eval.data_dim,
                            save_to=os.path.join(plot_folder_name, f'level_curves_potential_t_{t}') if save_locally else None
                        )
                        plt.close(level_curves_potential_fig)

                level_curves_interaction_fig = plot_level_curves(
                    interaction, ((-4, -4), (4, 4)), dimensions=dataset_eval.data_dim,
                    save_to=os.path.join(plot_folder_name, 'level_curves_interaction') if save_locally else None
                )

                if args.wandb:
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
            wandb_logs = {
                'epoch': epoch,
            }
            if config['metrics']['w_one_ahead']:
                error_w_one_ahead = dataset_eval.error_wasserstein_one_step_ahead(
                    potential,
                    beta,
                    interaction,
                    key_eval,
                    model=str(args.solver),
                    plot_folder_name=plot_folder_name if save_locally else None
                )
                print(f"Epoch {epoch} | Wasserstein error one step ahead: {error_w_one_ahead}, aggregate: {jnp.mean(error_w_one_ahead)} +/- {jnp.std(error_w_one_ahead)}")
                wandb_logs['error_w_one_ahead'] = float(jnp.mean(error_w_one_ahead))
                wandb_logs['error_w_one_ahead_std'] = float(jnp.std(error_w_one_ahead))
            if config['metrics']['w_cumulative']:
                error_w_cumulative = dataset_eval.error_wasserstein_cumulative(
                    predictions,
                    model=str(args.solver),
                    plot_folder_name=plot_folder_name if save_locally else None
                )
                print(f"Epoch {epoch} | Wasserstein error cumulative: {error_w_cumulative}")
                wandb_logs['error_w_cumulative'] = float(error_w_cumulative[-1])

            if args.wandb:
                wandb.log(wandb_logs)

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
        help=f"""Name of the dataset to train the model on. The name of the dataset should match the name of the directory generated by the `data_generator.py` script.""",
        )

    parser.add_argument(
        '--eval',
        type=str,
        default='test_data',
        choices=['train_data', 'test_data'],
        help=f"""Option to test fit on test data or train data (e.g., for debugging purposes).""",
    )
    
    parser.add_argument('--wandb', action='store_true',
                        help='Option to run with activated wandb.')

    parser.add_argument('--debug', action='store_true',
                        help='Option to run in debug mode.')
    
    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model.')
    
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