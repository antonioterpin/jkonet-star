"""
Module for handling datasets and computing prediction errors in population dynamics.

This module provides several dataset classes designed for loading and accessing different formats of the population trajectory data, including trajectory data and coupling data, and testing fit and prediction errors.

Classes
-------
    - ``PopulationDataset``
        Handles loading and batching of particle trajectory data. The single unit if a particle trajectory.
    - ``CouplingsDataset``
        Loads coupling data for trajectory models, including weights, features, and densities. The single unit is a coupling.
    - ``LinearParametrizationDataset``
        Loads data for the linear parametrization. The single unit is the entire dataset.
    - ``PopulationEvalDataset``
        Facilitates evaluation of model predictions using particle trajectories and computes prediction errors such as the Wasserstein distance.
"""
import glob
import os
import math
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from utils.ot import wasserstein_loss
from utils.sde_simulator import get_SDE_predictions
from utils.plotting import plot_predictions

from collections import defaultdict
from typing import Tuple, Optional, Callable, List


class PopulationDataset(Dataset):
    """
    Dataset class for loading and accessing particle trajectory data.

    The dataset is expected to be located in a directory named 'data/{dataset_name}' and consist of a single .npy file named 'data.npy'. The data contains particle trajectories over time, where each timestep has a set of particles.

    If the number of particles in a timestep is less than the maximum number of particles in any timestep, the dataset wraps around to handle the imbalance.

    Attributes
    ----------
    trajectory : np.ndarray
        Array of shape (num_timesteps, num_particles, num_features) containing
        the particle trajectories. Each entry in the array represents a particle's
        state at a given timestep.
    """
    def __init__(self, dataset_name: str, batch_size: int) -> None:
        """
        Initialize the PopulationDataset by loading data from 'data.npy'.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load. The dataset should be located in
            'data/{dataset_name}' and should contain a .npy file named 'data.npy'.
        """
        self.data = np.load(os.path.join('data', dataset_name, 'data.npy'))
        self.sample_labels = np.load(os.path.join('data', dataset_name, 'sample_labels.npy'))
        self.batch_size = batch_size

        # Group particles by their timestep using a defaultdict
        self.trajectory = defaultdict(list)
        for value, label in zip(self.data, self.sample_labels):
            self.trajectory[label].append(value)
        # Convert lists to numpy arrays
        for label in self.trajectory:
            self.trajectory[label] = np.array(self.trajectory[label])

        # Find the maximum number of particles in any timestep
        self.max_particles = max([particles.shape[0] for particles in self.trajectory.values()])
        if self.max_particles % self.batch_size != 0:
            self.max_particles = math.ceil(self.max_particles / self.batch_size) * self.batch_size

    def __len__(self) -> int:
        """
        Returns the number of timesteps in the dataset.

        Returns
        -------
        int
            The number of timesteps in the dataset.
        """

        return self.max_particles
    def __getitem__(self, idx: int) -> list:
        """
        Retrieve particle data for each timestep at the given index.

        Parameters
        ----------
        idx : int
            The index of the particle to retrieve.

        Returns
        -------
        list of np.ndarray
            A list where each element is an array representing the state of a
            particle at each timestep. The length of the list corresponds to the
            number of timesteps, and each array represents the particle state
            at a specific timestep.
        """
        particle_index = idx % self.max_particles
        # Retrieve the state of this particle index for each timestep
        # Wrapping so to handle unbalanced number of particles in each timestep
        # as if we were sampling
        return [self.trajectory[timestep][particle_index % len(self.trajectory[timestep])]
                for timestep in sorted(self.trajectory.keys())]


class CouplingsDataset(Dataset):
    """
    Dataset class for loading and accessing couplings data.

    The dataset is expected to be located in a directory named 'data/{dataset_name}' and consist of multiple .npy files. It provides access to input features, target features, time labels, weights, density values, and density gradients.

    Attributes
    ----------
    weight : np.ndarray
        Array of weights extracted from the couplings data.
    x : np.ndarray
        Array of input features extracted from the couplings data.
    y : np.ndarray
        Array of target features extracted from the couplings data.
    time : np.ndarray
        Array of time labels extracted from the couplings data.
    densities : np.ndarray
        Array of density values extracted from the densities files.
    densities_grads : np.ndarray
        Array of gradients of densities extracted from the densities files.
    """
    def __init__(self, dataset_name: str) -> None:
        """
        Initialize the CouplingsDataset by loading data from .npy files.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load. The dataset is expected to be located in a
            directory named 'data/{dataset_name}' and consist of multiple .npy files.
        """
        # load couplings for all timesteps together
        couplings = np.concatenate([np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'couplings_*.npy'))])
        self.weight = couplings[:, -1]
        self.x = couplings[:, :(couplings.shape[1] - 2) // 2]
        self.y = couplings[:, (couplings.shape[1] - 2) // 2:-2]
        self.time = couplings[:, -2]
        self.densities = np.concatenate(
            [np.load(f) for f in glob.glob(
                os.path.join('data', dataset_name, 'density_and_grads_*.npy'))]
        )
        self.densities_grads = self.densities[:, 1:]
        self.densities = self.densities[:, 0]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Retrieve a sample (x, y, t, w, rho, rho_grad) from the dataset at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            A tuple containing:

            - Input features (jnp.ndarray): Initial particle distribution.
            - Target features (jnp.ndarray): Target particle distribution.
            - Time label (jnp.ndarray): Time label.
            - Weight of the coupling (jnp.ndarray): Weight of the coupling.
            - Density value (jnp.ndarray): Density value.
            - Gradient of densities (jnp.ndarray): Gradient of densities.
        """
        return self.x[idx], self.y[idx], self.time[idx], self.weight[idx], self.densities[idx], self.densities_grads[
            idx]
    
class LinearParametrizationDataset(Dataset):
    """
    This dataset class loads and organizes data necessary for linear parametrization solver tasks, for which all data is analyzed together.
    """
    def __init__(self, dataset_name: str) -> None:
        """
        Initialize the LinearParametrizationDataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load.

        """
        couplings = [np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'couplings_*.npy'))]

        densities = [np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'density_and_grads_*.npy'))]
        self.data = [(
            c[:, :(c.shape[1] - 1) // 2], 
            c[:, (c.shape[1] - 1) // 2:-2],
            c[:, -2],
            c[:, -1],
            densities[t][:,0],
            densities[t][:,1:]
        ) for t, c in enumerate(couplings)]
        
    def __len__(self) -> int:
        """
        Return the number of elements in the dataset.

        Returns
        -------
        int
            The number of elements (always 1 for this dataset).
        """
        return 1
    
    def __getitem__(self, _)-> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieve the entire dataset.

        Since for the linear parametrization all data is used together, this method returns all data at once and the index parameter `_` is ignored.

        Parameters
        ----------
        _ : any
            This parameter is ignored.

        Returns
        -------
        List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]
            A list of tuples, where each tuple contains:

        - Input features (jnp.ndarray): Initial particle distribution.
        - Target features (jnp.ndarray): Target particle distribution.
        - Time label (jnp.ndarray): Time label associated with each sample.
        - Weight of the coupling (jnp.ndarray): Weight assigned to the coupling.
        - Density values (jnp.ndarray): Density values.
        - Gradient of densities (jnp.ndarray): Gradient of the density values.
        """
        return self.data
    
class PopulationEvalDataset(Dataset):
    """
    This dataset class loads and organizes population trajectory data for evaluation.
    
    Attributes
    ----------
    trajectory : dict
        A dictionary where each key corresponds to a unique timestep in the dataset, and the value is an array of trajectory data associated with that timestep.
    label_mapping : dict
        A dictionary mapping the original sample labels to consecutive integer indices.
    T : int
        The number of timesteps in the trajectories.
    data_dim : int
        The dimensionality of the data at each timestep.
    no_ground_truth : bool
        Flag indicating if the dataset lacks a ground truth file.
    potential : str
        The potential function used in the predictions.
    internal : str
        The internal dynamics setting used.
    beta : float
        The beta parameter used in the simulations.
    interaction : str
        The interaction function used in the predictions.
    dt : float
        The timestep size used in the simulation.
    trajectory_only_potential : np.ndarray
        Trajectory predictions considering only the potential term.
    trajectory_only_interaction : np.ndarray
        Trajectory predictions considering only the interaction term.
    """
    def __init__(self, key, dataset_name: str, solver: str, wasserstein_metric: int, label='test_data'):
        """
        Initialize the PopulationEvalDataset.

        Parameters
        ----------
        key : Any
            A key used for random number generation or seeding.
        dataset_name : str
            The name of the dataset to load. The data should be located in the directory
            'data/{dataset_name}' and consist of .npy files.
        solver : str
            The solver method used, primarily for plotting or prediction purposes.
        wasserstein_metric: int
            Specifies the order of the Wasserstein distance to be used for the error calculation.
        label : str, optional
            Specifies whether to load 'test_data' or 'train_data'. Default is 'test_data'.

        """
        # dt does not actually matter for learning, 
        # because everything can be scaled accordingly - as long as it 
        # is always consistent
        self.dt: float = 1.0
        self.key = key
        self.solver = solver
        self.wasserstein_metric = wasserstein_metric
        if label == 'test_data':
            data = np.load(os.path.join('data', dataset_name, 'test_data.npy'))
            sample_labels = np.load(os.path.join('data', dataset_name, 'test_sample_labels.npy'))
        else:
            data = np.load(os.path.join('data', dataset_name, 'train_data.npy'))
            sample_labels = np.load(os.path.join('data', dataset_name, 'train_sample_labels.npy'))

        unique_labels = np.unique(sample_labels)
        self.label_mapping = {original: i for i, original in enumerate(unique_labels)}

        self.trajectory = defaultdict(list)
        for value, label in zip(data, sample_labels):
            self.trajectory[self.label_mapping[label]].append(value)
        for label in self.trajectory:
            self.trajectory[label] = np.array(self.trajectory[label])
            self.data_dim = self.trajectory[label].shape[1]
        self.T = len(self.trajectory.keys())-1
        self.no_ground_truth = False

    def __len__(self) -> int:
        """
        Get the number of particles at the first timestep.

        Returns
        -------
        int
            The number of particles at the first timestep.
        """
        return self.trajectory[0].shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieves a particle's features at the first timestep.

        Parameters
        ----------
        idx : int
            The index of the particle to retrieve.

        Returns
        -------
        np.ndarray
            The features of the specified particle at the first timestep.
        """
        return self.trajectory[0][idx, :]

    def error_wasserstein(self, trajectory_predicted: np.ndarray) -> float:
        """
        Compute the Wasserstein loss between the predicted and true trajectories.

        This method calculates the Wasserstein distance (a measure of distance
        between probability distributions) between the predicted trajectories
        and the true trajectories over all timesteps.

        Parameters
        ----------
        trajectory_predicted : np.ndarray
            The predicted trajectory with shape (T, n_particles, n_features).

        Returns
        -------
        float
            The cumulative Wasserstein error over all timesteps.
        """
        error = 0
        for t in range(1, trajectory_predicted.shape[0]):
            error += wasserstein_loss(
                        trajectory_predicted[t], jnp.asarray(self.trajectory[t]), self.wasserstein_metric)
        return error

    def error_wasserstein_one_step_ahead(
            self,
            potential: Callable[[jnp.ndarray], float],
            beta: float,
            interaction: Callable[[jnp.ndarray], float],
            key_eval: jnp.ndarray,
            model: str,
            plot_folder_name: Optional[str] = None
    ) -> jnp.ndarray:
        """
        Compute the Wasserstein error for one-step-ahead predictions.

        This method evaluates the prediction error by computing the Wasserstein distance between the predicted trajectory and the actual trajectory at each timestep, given the current true population.

        Parameters
        ----------
        potential : Callable[[jnp.ndarray], float]
            Function that computes the potential based on a JAX array input.
        beta : float
            Beta parameter used in the predictions.
        interaction : Callable[[jnp.ndarray], float]
            Function that computes the interaction based on a JAX array input.
        key_eval : jnp.ndarray
            Random key for JAX-based random number generation.
        model : str
            Name of the solver model used. This is primarily used for plotting purposes.
        plot_folder_name : Optional[str], default=None
            Directory path where plots should be saved. If None, no plots will be saved.


        Returns
        -------
        jnp.ndarray
            An array of Wasserstein errors for the one-step-ahead predictions over timesteps.
            The array has length `T`, where each entry corresponds to the error at a specific
            timestep.
        """
        error_wasserstein_one_ahead = jnp.ones(self.T)
        for t in range(self.T):
            init = self.trajectory[t]
            predictions = get_SDE_predictions(
                    self.solver,
                    self.dt,
                    1,
                    t+1,
                    potential,
                    beta,
                    interaction,
                    key_eval,
                    init)
            if plot_folder_name:
                plot_filename = f'one_ahead_tp_{t + 1}'
                plot_path = os.path.join(plot_folder_name, plot_filename)
                prediction_fig = plot_predictions(
                    predictions[-1].reshape(1, -1, self.data_dim),
                    self.trajectory,
                    interval=(t + 1, t + 1),
                    model=model,
                    save_to=plot_path)
                plt.close(prediction_fig)
            error_wasserstein_one_ahead = error_wasserstein_one_ahead.at[t].set(
                wasserstein_loss(predictions[-1], jnp.asarray(self.trajectory[t + 1]), self.wasserstein_metric))
        return error_wasserstein_one_ahead

    def error_wasserstein_cumulative(
        self,
        predictions: jnp.ndarray,
        model: str,
        plot_folder_name: Optional[str] = None
    ) -> jnp.ndarray:
        """
        Compute the cumulative Wasserstein error per timestep.

        This method calculates the Wasserstein distance between the predicted and actual
        trajectories at each timestep and returns the cumulative error.

        Parameters
        ----------
        predictions : jnp.ndarray
            Array of predicted trajectories with shape (T+1, n_particles, n_features).
            The predictions should cover the entire timespan from 0 to T.

        model : str
            Name of the solver model used. This is primarily used for plotting purposes.

        plot_folder_name : Optional[str], default=None
            Directory path where plots should be saved. If None, no plots will be saved.

        Returns
        -------
        jnp.ndarray
            Array of cumulative Wasserstein errors, with each entry corresponding to
            the error at a specific timestep. The array has length `T`.
        """
        error_wasserstein_cumulative = jnp.ones(self.T)
        for t in range(1, self.T + 1):
            if plot_folder_name:
                plot_path = os.path.join(plot_folder_name, f'cum_tp_{t}')
                trajectory_fig = plot_predictions(
                    predictions[t].reshape(1, -1, self.data_dim),
                    self.trajectory,
                    interval=(t, t),
                    model=model,
                    save_to=plot_path)
                plt.close(trajectory_fig)
            error_wasserstein_cumulative = error_wasserstein_cumulative.at[t - 1].set(
                wasserstein_loss(predictions[t], jnp.asarray(self.trajectory[t]), self.wasserstein_metric))
        return error_wasserstein_cumulative