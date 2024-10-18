"""
Module for generating and processing population trajectory data.

This module provides tools to:

1. Simulate particle trajectories with different potential, internal and interaction energy configurations, or load pre-existing data.
2. Fit Gaussian Mixture Models (GMMs) on trajectory data.
3. Compute couplings between particle distributions at consecutive timesteps.
4. Generate density and gradient data from the simulated or loaded trajectories.
5. Plot couplings, density levels, and particle trajectories.

Steps 2-4 are the preprocessing steps required to train a JKOnet* model.

Functions
---------
- ``filename_from_args``
    Generates a descriptive filename based on the provided command-line arguments.
    
- ``train_test_split``
    Splits a dataset into training and testing subsets, ensuring the label distribution is preserved.
    
- ``generate_data_from_trajectory``
    Processes trajectory data by fitting a GMM, computing couplings, saving data, and plotting particle densities and couplings.
    
- ``main``
    Main entry point for the data generation pipeline. It handles argument parsing, SDE simulation, train-test splitting, and calls functions to generate and save the processed data.

Example
-------
To generate synthetic trajectory data with 1000 particles, a chosen potential, and internal Wiener energy:

.. code-block:: bash

    python data_generator.py --n-particles 1000 --potential styblinski_tang --n-timesteps 5

To load previously generated data and compute couplings:

    python data_generator.py --load-from-file my_trajectory_data --test-ratio 0.2 --n-gmm-components 5

Command-line Arguments
----------------------
The script accepts the following command-line arguments:

- `--load-from-file` (`str`):
    Load trajectory data from a file instead of generating it. Must be a NumPy array of shape `(n_timesteps + 1, n_particles, dimension)`.

- `--potential` (`str`):
    Specify the potential energy to use for the SDE simulation.

- `--n-timesteps` (`int`):
    Number of timesteps for the SDE simulation.

- `--dt` (`float`):
    Timestep size for the SDE simulation.

- `--internal` (`str`):
    Type of internal energy (e.g., `'wiener'`) to use in the simulation.

- `--beta` (`float`):
    Standard deviation of the Wiener process for internal energy.

- `--interaction` (`str`):
    Specify the interaction energy between particles.

- `--dimension` (`int`):
    Dimensionality of the simulated system.

- `--n-particles` (`int`):
    Number of particles in the simulation.

- `--batch-size` (`int`):
    Batch size for computing couplings during the data processing phase.

- `--n-gmm-components` (`int`):
    Number of components for the Gaussian Mixture Model.

- `--seed` (`int`):
    Random seed for reproducibility.

- `--test-ratio` (`float`):
    Proportion of data to be used as test data during splitting.

- `--split-population` (`bool`):
    If set, data is split at every timestep; otherwise, it is split along the trajectories.

- `--leave-one-out` (`int`):
    If non-negative, leaves one time point out from the training set.

- `--sinkhorn` (`float`):
    Regularization parameter for the Sinkhorn algorithm. If < 1e-12, no regularization is applied.

- `--dataset-name` (`str`):
    Specifies the name of the output dataset. If not provided, a directory name will be automatically generated based on the simulation parameters. This option is only used if data is generated. If data is loaded from a file (using `--load-from-file`), the output dataset will retain the name of the input file.
"""


import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
# import matplotlib.pyplot as plt
from utils.functions import potentials_all, interactions_all
from utils.sde_simulator import SDESimulator
from utils.density import GaussianMixtureModel
from utils.ot import compute_couplings, compute_couplings_sinkhorn
from utils.plotting import plot_level_curves
from collections import defaultdict
from typing import Tuple
import time

def filename_from_args(args):
    """
    Generates a filename based on the arguments given.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the command line. See main() for the arguments.

    Returns
    -------
    str
        Generated filename based on the provided arguments.
    """

    # Use dataset name if provided, else generate filename from args
    if args.dataset_name:
        return args.dataset_name

    # Generate filename
    filename = f"potential_{args.potential}_"
    filename += f"internal_{args.internal}_"
    filename += f"beta_{args.beta}_"
    filename += f"interaction_{args.interaction}_"
    filename += f"dt_{args.dt}_"
    filename += f"T_{args.n_timesteps}_"
    filename += f"dim_{args.dimension}_"
    filename += f"N_{args.n_particles}_"
    filename += f"gmm_{args.n_gmm_components}_"
    filename += f"seed_{args.seed}_"
    filename += f"split_{args.test_ratio}"
    filename += f"_split_trajectories_{not args.split_population}"
    filename += f"_lo_{args.leave_one_out}"
    filename += f"_sinkhorn_{args.sinkhorn}"
    
    return filename

def train_test_split(
    values: jnp.ndarray,
    sample_labels: jnp.ndarray,
    test_ratio: float = 0.4,
    split_trajectories: bool = True,
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Splits the dataset into training and testing sets while preserving the distribution of labels.

    This function ensures that the proportion of each label in the dataset is preserved in both the
    training and testing subsets.

    Parameters
    ----------
    values : jnp.ndarray
        The data array to be split.
    sample_labels : jnp.ndarray
        The corresponding labels for the data. Contains the timestep
        linked to each value.
    test_ratio : float, optional
        The proportion of the dataset to include in the test split. Defaults to 0.4.
    split_trajectories : bool, optional
        If True, the data is split by trajectories. Defaults to True.
        If False, the data is split by individual data points.
    seed : int, optional
        Random seed for reproducibility. Defaults to 0.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        A tuple containing:

        - Train values: Subset of the data for training.
        - Train labels: Corresponding labels for the training data.
        - Test values: Subset of the data for testing.
        - Test labels: Corresponding labels for the testing data.
    """
    np.random.seed(seed)
    #Check if the dataset is balanced
    unique_labels, counts = np.unique(sample_labels, return_counts=True)
    is_balanced = np.all(counts == counts[0])

    assert (not split_trajectories) or is_balanced, "Trajectories are not balanced, cannot split by trajectories."

    if split_trajectories:
        n_particles = counts[0]
        indices = np.arange(n_particles)
        np.random.shuffle(indices)
        test_size = int(n_particles * test_ratio)

        train_indices_block = indices[:-test_size]
        test_indices_block = indices[-test_size:]

        train_indices = []
        test_indices = []

        # For each block, apply the same test/train split by adding block-size offsets
        for label in unique_labels:
            offset = label * n_particles
            train_indices.extend(train_indices_block + offset)
            test_indices.extend(test_indices_block + offset)

    else:
        unique_labels = np.unique(sample_labels)
        train_indices = []
        test_indices = []

        for label in unique_labels:
            indices = np.where(sample_labels == label)[0]
            np.random.shuffle(indices)
            split = int(len(indices) * (1 - test_ratio))
            train_indices.extend(indices[:split])
            test_indices.extend(indices[split:])


    return values[np.array(train_indices)], sample_labels[np.array(train_indices)], values[np.array(test_indices)], sample_labels[np.array(test_indices)]



def generate_data_from_trajectory(
        folder: str,
        values: jnp.ndarray,
        sample_labels: jnp.ndarray,
        n_gmm_components: int = 10,
        batch_size: int = 1000,
        leave_one_out: int = -1,
        sinkhorn: float = 0.0
) -> None:
    """
    Preprocesses the trajectory data for JKOnet*.

    Fits Gaussian Mixture Models (GMM) to the trajectory data, computes couplings, and saves the results to disk. This function also plots the data and saves the plots.

    Parameters
    ----------
    folder : str
        Directory where the data and plots will be saved.
    values : jnp.ndarray
        Array of trajectory data points.
    sample_labels : jnp.ndarray
        Array of sample labels corresponding to each data point.
    n_gmm_components : int, optional
        Number of components for the Gaussian Mixture Model (default is 10).
    batch_size : int, optional
        Batch size for computing couplings (default is 1000).
    leave_one_out : int, optional
        If non-negative, leaves one time point out from the training set (default is -1).
    sinkhorn : float, optional
        Regularization parameter for the Sinkhorn algorithm. If < 1e-12, no regularization is applied (default is 0.0).

    Returns
    -------
    None
    """
    sample_labels = [int(label) for label in sample_labels]
    # Group the values by sample labels
    trajectory = defaultdict(list)
    for value, label in zip(values, sample_labels):
        trajectory[label].append(value)

    # Convert lists to arrays
    trajectory = {label: jnp.array(values) for label, values in trajectory.items()}
    sorted_labels = sorted(trajectory.keys())

    # Check if the dataset is unbalanced (i.e., varying number of particles at each timestep)
    num_particles_per_step = [trajectory[label].shape[0] for label in sorted_labels]
    is_unbalanced = len(set(num_particles_per_step)) > 1

    if n_gmm_components > 0:
        print("Fitting Gaussian Mixture Model...")
        gmm = GaussianMixtureModel()
        gmm.fit(trajectory, n_gmm_components, args.seed)
        # cmap = plt.get_cmap('Spectral')

        # all_values = jnp.vstack([trajectory[label] for label in sorted_labels])
        # x_min = jnp.min(all_values[:, 0]) * 0.9
        # x_max = jnp.max(all_values[:, 0]) * 1.1
        # y_min = jnp.min(all_values[:, 1]) * 0.9
        # y_max = jnp.max(all_values[:, 1]) * 1.1

        # for label in sorted_labels:
        #     # Plot particles
        #     plt.scatter(trajectory[label][:, 0], trajectory[label][:, 1],
        #                 c=[cmap(float(label) / len(sorted_labels))], marker='o', s=4)
        #     plt.xlim(x_min, x_max)
        #     plt.ylim(y_min, y_max)
        #     plt.savefig(os.path.join('out', 'plots', folder, f'density_{label}.png'))
        #     plt.clf()

    print("Computing couplings...")

    if sinkhorn > 1e-12:
        # change compute_couplings to use the sinkhorn function
        f_compute_couplings = lambda x, y, t: compute_couplings_sinkhorn(x, y, t, sinkhorn)
    else:
        f_compute_couplings = lambda x, y, t: compute_couplings(x, y, t)

    for t, label in enumerate(sorted_labels[:-1]):
        if leave_one_out == t or leave_one_out == t + 1:
            continue
        next_label = sorted_labels[t + 1]
        values_t = trajectory[label]
        values_t1 = trajectory[next_label]

        # Compute couplings
        time_t = time.time()
        if is_unbalanced or batch_size < 0:
            couplings = f_compute_couplings(
                values_t,
                values_t1,
                next_label)
        else:
            couplings = []
            for i in range(int(jnp.ceil(trajectory[0].shape[0]/ batch_size))):
                idxs = jnp.arange(i * batch_size, min(
                    trajectory[0].shape[0], (i + 1) * batch_size
                ))
                couplings.append(f_compute_couplings(
                    trajectory[t][idxs, :],
                    trajectory[t + 1][idxs, :],
                    next_label
                ))
            couplings = jnp.concatenate(couplings, axis=0)
        time_couplings = time.time() - time_t
        print(f"Time to compute couplings: {time_couplings} [s]")
        jnp.save(os.path.join('data', folder, f'couplings_{label}_to_{next_label}.npy'), couplings)
        # Save densities and gradients
        ys = couplings[:, (couplings.shape[1] - 1) // 2:-2] #Changed the 2 to match the new shape of couplings
        rho = lambda _: 0.
        if n_gmm_components > 0:
            rho = lambda x: gmm.gmm_density(t+1, x)
        densities = jax.vmap(rho)(ys).reshape(-1, 1)
        densities_grads = jax.vmap(jax.grad(rho))(ys)
        data = jnp.concatenate([densities, densities_grads], axis=1)
        jax.numpy.save(os.path.join('data', folder, f'density_and_grads_{label}_to_{next_label}.npy'), data)

        # # Plot couplings
        # plot_couplings(couplings)
        # plt.xlim(x_min, x_max)
        # plt.ylim(y_min, y_max)
        # plt.savefig(os.path.join('out', 'plots', folder, f'couplings_{label}_to_{next_label}.png'))
        # plt.clf()

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the data generation and processing pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments (see the module docstring for details).

    Returns
    -------
    None
    """
    print("Running with arguments: ", args)
    key = jax.random.PRNGKey(args.seed)

    folder = filename_from_args(args) if args.load_from_file is None else args.load_from_file

    if not os.path.exists(os.path.join('data', folder)):
        os.makedirs(os.path.join('data', folder))
    if not os.path.exists(os.path.join('out', 'plots', folder)):
        os.makedirs(os.path.join('out', 'plots', folder))

    if args.load_from_file is None:
        sde_simulator = SDESimulator(
            args.dt,
            args.n_timesteps,
            1,
            potentials_all[args.potential] if args.potential != 'none' else False,
            args.beta if args.internal == 'wiener' else False,
            interactions_all[args.interaction] if args.interaction != 'none' else False
        )
        print("Generating data...")
        init_pp = jax.random.uniform(
            key,
            (args.n_particles, args.dimension), minval=-4, maxval=4)
        trajectory = sde_simulator.forward_sampling(key, init_pp)

        data = trajectory.reshape(trajectory.shape[0] * trajectory.shape[1], trajectory.shape[2])
        sample_labels = jnp.repeat(jnp.arange(args.n_timesteps+1), trajectory.shape[1])
        jax.numpy.save(os.path.join('data', folder, 'data.npy'), data)
        jax.numpy.save(os.path.join('data', folder, "sample_labels.npy"), sample_labels)

        # Save args to file
        with open(os.path.join('data', folder, 'args.txt'), 'w') as file:
            file.write(f"potential={args.potential}\n")
            file.write(f"internal={args.internal}\n")
            file.write(f"beta={args.beta}\n")
            file.write(f"interaction={args.interaction}\n")
            file.write(f"dt={args.dt}\n")

        if args.potential != 'none':
            potential = potentials_all[args.potential]
            plot_level_curves(potential, ((-4, -4), (4, 4)),
                              save_to=os.path.join('out', 'plots', folder, 'level_curves_potential'))
        if args.interaction != 'none':
            interaction = interactions_all[args.interaction]
            plot_level_curves(interaction, ((-4, -4), (4, 4)),
                              save_to=os.path.join('out', 'plots', folder, 'level_curves_interaction'))
    else:
        print("Loading data from file...")
        folder = args.load_from_file
        data = jax.numpy.load(os.path.join('data', folder, 'data.npy'))
        sample_labels = jax.numpy.load(os.path.join('data', folder, 'sample_labels.npy'))

    # Perform train-test split
    assert args.test_ratio >= 0 and args.test_ratio <= 1, "Test split must be a proportion."
    if args.test_ratio > 0:
        train_values, train_labels, test_values, test_labels = train_test_split(
            data,
            sample_labels,
            args.test_ratio,
            not args.split_population,
            args.seed)
    else:
        train_values, train_labels = data, sample_labels

    # Generate data for train set
    jax.numpy.save(os.path.join('data', folder, 'train_data.npy'), train_values)
    jax.numpy.save(os.path.join('data', folder, 'train_sample_labels.npy'), train_labels)
    generate_data_from_trajectory(
        folder, train_values, train_labels,
        args.n_gmm_components, args.batch_size,
        args.leave_one_out,
        args.sinkhorn)

    if args.test_ratio > 0:
        # Generate data for test set
        jax.numpy.save(os.path.join('data', folder, 'test_data.npy'), test_values)
        jax.numpy.save(os.path.join('data', folder, 'test_sample_labels.npy'), test_labels)

    print("Done.")


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load-from-file', 
        type=str, 
        default=None, 
        help="""
        Instead of generating a synthetic trajectory, load it from a file.

        The trajectory must be a numpy array of shape (n_timesteps + 1, n_particles, dimension).
        """
    )
    
    parser.add_argument(
        '--potential', 
        type=str, 
        default='none',
        choices=list(potentials_all.keys()) + ['none'],
        help="""Name of the potential energy to use."""
        )
    
    parser.add_argument(
        '--n-timesteps', 
        type=int, 
        default=5,
        help="""Number of timesteps of the simulation of the SDE."""
        )
    
    parser.add_argument(
        '--dt', 
        type=float, 
        default=0.01,
        help="""dt in the simulation of the SDE."""
        )
    
    parser.add_argument(
        '--internal', 
        type=str, 
        default='none',
        choices=['wiener', 'none'],
        help="""Name of the internal energy to use.
        
        Note: 

            - `'wiener'` requires additionally the ``--beta`` parameter.
            - `'none'` means no internal energy is considered.
            - At the moment only the wiener process is implemented.
        """
        )
    
    parser.add_argument(
        '--beta', 
        type=float, 
        default=0.0,
        help="""Standard deviation of the wiener process. Must be positive.
        
        Note: This parameter is considered only if ``--internal`` is `'wiener'`.
        """
        )
    
    parser.add_argument(
        '--interaction', 
        type=str, 
        default='none',
        choices=list(interactions_all.keys()) + ['none'],
        help="""
        Name of the interaction energy to use, `'none'` means no interaction energy is considered.
        """
        )
    
    parser.add_argument(
        '--dimension', 
        type=int, 
        default=2,
        help="""
        Dimensionality of the particles generated in the synthetic data.
        """
        )
    
    parser.add_argument(
        '--n-particles', 
        type=int, 
        default=2000,
        help="""
        Number of particles sampled generated.
        """
        )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for computing the couplings. Negative values mean no batching.'
    )
    
    parser.add_argument(
        '--n-gmm-components',
        type=int,
        default=10,
        help='Number of components of the Gaussian Mixture Model. 0 for no GMM.'
    )
    
    # reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Set seed for the run.'
    )

    # Train-test split
    parser.add_argument(
        '--test-ratio',
        type=float, 
        default=0.5,
        help='Ratio of the data allocated to the test set.'
        )

    # Flag to perform splitting on trajectories
    parser.add_argument(
        '--split-population',
        action='store_true',
        help='If set, data is split at every timestep. If not set, it is split along trajectories.'
    )

    # Leave one time-point out
    parser.add_argument(
        '--leave-one-out',
        type=int,
        default=-1,
        help='If non-negative, leaves one-time point out from the training set.'
    )

    parser.add_argument(
        '--sinkhorn',
        type=float,
        default=0.0,
        help='Regularization parameter for the Sinkhorn algorithm. If < 1e-12, no regularization is applied.'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Name for the dataset.')

    args = parser.parse_args()

    main(args)
