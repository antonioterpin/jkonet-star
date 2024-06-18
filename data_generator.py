import os
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils.functions import potentials_all, interactions_all
from utils.sde_simulator import SDESimulator
from utils.density import GaussianMixtureModel
from utils.ot import compute_couplings
from utils.plotting import plot_couplings, plot_level_curves

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

def generate_data_from_trajectory(folder, trajectory, n_gmm_components=10, batch_size=1000):
    """
    Fits the gaussians and computes the couplings from the trajectory.

    - it saves the data to file
    - it fits a Gaussian Mixture Model to the data and saves the densities and gradients at the particles to file
    - computes the couplings between the particles and saves them to file
    - it plots all the information and saves the plots to file
    """
    if n_gmm_components > 0:
        print("Fitting Gaussian Mixture Model...")
        gmm = GaussianMixtureModel()
        gmm.fit(trajectory, args.n_gmm_components)
        # gmm.to_file(f"data/{filename}_gmm.pkl")
        for t in range(1, trajectory.shape[0]):
            # plot particles
            plt.scatter(trajectory[t, :, 0], trajectory[t, :, 1], c='blue', marker='o')
            plt.savefig(os.path.join('out', 'plots', folder, f'density_{t}.png'))
            plt.clf()

    print("Computing couplings...")
    for t in range(1, trajectory.shape[0]):
        couplings = []
        for i in range(int(jnp.ceil(trajectory.shape[1] / batch_size))):
            idxs = jnp.arange(i * batch_size, min(
                trajectory.shape[1], (i + 1) * batch_size
            ))
            couplings.append(compute_couplings(
                trajectory[t - 1, idxs, :], 
                trajectory[t, idxs, :]
            ))
        couplings = jnp.concatenate(couplings, axis=0)
        jax.numpy.save(os.path.join('data', folder, f'couplings_{t}.npy'), couplings)
        # save densities and grads
        ys = couplings[:, (couplings.shape[1] - 1) // 2:-1]
        rho = lambda x: 0.
        if n_gmm_components > 0:
            rho = lambda x: gmm.gmm_density(t, x)
        densities = jax.vmap(rho)(ys).reshape(-1, 1)
        densities_grads = jax.vmap(jax.grad(rho))(ys)
        data = jnp.concatenate([densities, densities_grads], axis=1)
        jax.numpy.save(os.path.join('data', folder, f'density_and_grads_{t}.npy'), data)
        plot_couplings(couplings)
        plt.savefig(os.path.join('out', 'plots', folder, f'couplings_{t}.png'))
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
        jax.numpy.save(os.path.join('data', folder, 'data.npy'), trajectory)

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
        trajectory = jax.numpy.load(os.path.join('data', folder, 'data.npy'))

    # From here, the methodology is the same as for non-synthetic data
    generate_data_from_trajectory(folder, trajectory, args.n_gmm_components, args.batch_size)

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

    args = parser.parse_args()

    main(args)
