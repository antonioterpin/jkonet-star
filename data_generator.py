import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from utils.functions import potentials_all, interactions_all
from utils.sde_simulator import SDESimulator
from utils.density import GaussianMixtureModel
from utils.ot import compute_couplings
from utils.plotting import plot_couplings, plot_level_curves
from collections import defaultdict

def filename_from_args(args):
    """
    Generates a filename based on the arguments given.

    Parameters:
        args (argparse.Namespace): Arguments parsed from the command line.
        - See main() for the arguments.
    """

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
    filename += f"seed_{args.seed}"
    
    return filename

def train_test_split(values, sample_labels, test_size=0.4):
    unique_labels = np.unique(sample_labels)
    train_indices = []
    test_indices = []

    for label in unique_labels:
        indices = np.where(sample_labels == label)[0]
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])

    train_indices = jnp.array(train_indices)
    test_indices = jnp.array(test_indices)

    return values[train_indices], sample_labels[train_indices], values[test_indices], sample_labels[test_indices]

def generate_data_from_trajectory(folder, values, sample_labels, n_gmm_components=10, batch_size=1000, data_type='train'):
    """
    Fits the gaussians and computes the couplings from the trajectory.

    - Saves the data to file
    - Fits a Gaussian Mixture Model to the data and saves the densities and gradients at the particles to file
    - Computes the couplings between the particles and saves them to file
    - Plots all the information and saves the plots to file
    """
    sample_labels = [int(label) for label in sample_labels]
    # Group the values by sample labels
    grouped_values = defaultdict(list)
    for value, label in zip(values, sample_labels):
        grouped_values[label].append(value)

    # Convert lists to arrays
    grouped_values = {label: jnp.array(values) for label, values in grouped_values.items()}
    sorted_labels = sorted(grouped_values.keys())

    if n_gmm_components > 0:
        print("Fitting Gaussian Mixture Model...")
        gmm = GaussianMixtureModel()
        gmm.fit(grouped_values, n_gmm_components)
        cmap = plt.get_cmap('Spectral')

        all_values = jnp.vstack([grouped_values[label] for label in sorted_labels])
        x_min = jnp.min(all_values[:, 0]) * 0.9
        x_max = jnp.max(all_values[:, 0]) * 1.1
        y_min = jnp.min(all_values[:, 1]) * 0.9
        y_max = jnp.max(all_values[:, 1]) * 1.1

        for label in sorted_labels:
            # Plot particles
            plt.scatter(grouped_values[label][:, 0], grouped_values[label][:, 1],
                        c=[cmap(float(label) / len(sorted_labels))], marker='o', s=4)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.savefig(os.path.join('out', 'plots', folder, f'density_{label}.png'))
            plt.clf()

    print("Computing couplings...")
    for t, label in enumerate(sorted_labels[:-1]):
        next_label = sorted_labels[t + 1]
        values_t = grouped_values[label]
        values_t1 = grouped_values[next_label]

        # Compute couplings
        couplings = compute_couplings(values_t, values_t1, next_label)
        jnp.save(os.path.join('data', folder, f'couplings_{data_type}_{label}_to_{next_label}.npy'), couplings)
        # Save densities and gradients
        ys = couplings[:, (couplings.shape[1] - 1) // 2:-2] #Changed the 2 to match the new shape of couplings
        rho = lambda x: 0.
        if n_gmm_components > 0:
            rho = lambda x: gmm.gmm_density(t, x)
        densities = jax.vmap(rho)(ys).reshape(-1, 1)
        densities_grads = jax.vmap(jax.grad(rho))(ys)
        data = jnp.concatenate([densities, densities_grads], axis=1)
        jax.numpy.save(os.path.join('data', folder, f'density_and_grads_{data_type}_{label}_to_{next_label}.npy'), data)
        
        # Plot couplings
        plot_couplings(couplings)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(os.path.join('out', 'plots', folder, f'couplings_{data_type}_{label}_to_{next_label}.png'))
        plt.clf()

def main(args):
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
    train_values, train_labels, test_values, test_labels = train_test_split(data, sample_labels, test_size=args.train_test_split)
    
    # Save train and test data
    jax.numpy.save(os.path.join('data', folder, 'train_data.npy'), train_values)
    jax.numpy.save(os.path.join('data', folder, 'train_sample_labels.npy'), train_labels)
    jax.numpy.save(os.path.join('data', folder, 'test_data.npy'), test_values)
    jax.numpy.save(os.path.join('data', folder, 'test_sample_labels.npy'), test_labels)

    # Generate data for train set
    generate_data_from_trajectory(folder, train_values, train_labels, args.n_gmm_components, args.batch_size, data_type='train')

    # Generate data for test set
    generate_data_from_trajectory(folder, test_values, test_labels, args.n_gmm_components, args.batch_size, data_type='test')

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
        help="""Name of the potential energy to use.
        
        Note: This parameter is considered only if --dataset is 'sde'.
        """
        )
    
    parser.add_argument(
        '--n-timesteps', 
        type=int, 
        default=5,
        help="""Number of timesteps of the simulation of the SDE.
        
        Note: This parameter is considered only if --dataset is 'sde'.
        """
        )
    
    parser.add_argument(
        '--dt', 
        type=float, 
        default=0.01,
        help="""dt in the simulation of the SDE.
        
        Note: This parameter is considered only if --dataset is 'sde'.
        """
        )
    
    parser.add_argument(
        '--internal', 
        type=str, 
        default='none',
        choices=['wiener', 'none'],
        help="""Name of the internal energy to use.
        
        Note: 
            - This parameter is considered only if --dataset is 'sde'.
            - 'wiener' requires additionally the --sd parameter.
            - 'none' means no internal energy is considered.
            - At the moment only wiener process is implemented.
        """
        )
    
    parser.add_argument(
        '--beta', 
        type=float, 
        default=0.0,
        help="""Standard deviation of the wiener process. Must be positive.
        
        Note: This parameter is considered only if --internal is 'wiener'.
        """
        )
    
    parser.add_argument(
        '--interaction', 
        type=str, 
        default='none',
        choices=list(interactions_all.keys()) + ['none'],
        help="""Name of the interaction energy to use.
        
        Note: 
            - This parameter is considered only if --dataset is 'sde'.
            - 'none' means no internal energy is considered.
        """
        )
    
    parser.add_argument(
        '--dimension', 
        type=int, 
        default=2,
        help="""
        Dimensionality of the system. Used to generate synthetic data.
        """
        )
    
    parser.add_argument(
        '--n-particles', 
        type=int, 
        default=1000,
        help="""
        Number of particles sampled generated.
        """
        )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for computing the couplings.'
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

    #Train-test split
    parser.add_argument(
        '--train-test-split', 
        type=float, 
        default=0.4,
        help="""Train test split.
        
        """
        )

    args = parser.parse_args()

    main(args)
