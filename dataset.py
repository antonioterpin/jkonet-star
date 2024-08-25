import glob
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from utils.functions import potentials_all, interactions_all
from utils.ot import wasserstein_loss
from utils.sde_simulator import get_SDE_predictions
from utils.plotting import plot_predictions

from collections import defaultdict
from typing import Tuple, Optional, Callable, List


class PopulationDataset(Dataset):
    def __init__(self, dataset_name: str):
        self.trajectory = np.load(os.path.join('data', dataset_name, 'data.npy'))

    def __len__(self):
        return self.trajectory.shape[1]

    def __getitem__(self, idx):
        # returns a particle for each timestep
        # batching means getting more particles per timestep
        return [self.trajectory[t, idx, :] 
                for t in range(self.trajectory.shape[0])]


class CouplingsDataset(Dataset):
    def __init__(self, dataset_name: str) -> None:
        """
        Dataset class for loading and accessing couplings data.

        Parameters:
        dataset_name (str): The name of the dataset to load. The dataset is expected to be located in a
                            directory named 'data/{dataset_name}' and consist of multiple .npy files.
        """
        # load couplings for all timesteps together
        couplings = np.concatenate([np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'couplings_train_*.npy'))])
        self.weight = couplings[:, -1]
        self.x = couplings[:, :(couplings.shape[1] - 2) // 2]
        self.y = couplings[:, (couplings.shape[1] - 2) // 2:-2]
        self.time = couplings[:, -2]
        self.densities = np.concatenate(
            [np.load(f) for f in glob.glob(
                os.path.join('data', dataset_name, 'density_and_grads_train_*.npy'))]
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

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.time[idx], self.weight[idx], self.densities[idx], self.densities_grads[
            idx]
    
class LinearParametrizationDataset(Dataset):
    def __init__(self, dataset_name: str) -> None:
        """
        LinearParametrizationDataset loads and organizes data for linear parametrization solver.

        Parameters:
        dataset_name (str): The name of the dataset to load. The data should be located in
                            'data/{dataset_name}' and consist of multiple .npy files.
        """
        couplings = [np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'couplings_train_*.npy'))]

        densities = [np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'density_and_grads_train_*.npy'))]
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
        Returns the number of elements in the dataset.

        Returns:
        int: The number of elements (always 1 for this dataset).
        """
        return 1
    
    def __getitem__(self, _)-> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieves the dataset.

        Parameters:
        _ (any): This parameter is ignored as the entire dataset is returned.

        Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            The entire dataset as a list of tuples, where each tuple contains:
            - Input features (np.ndarray)
            - Target features (np.ndarray)
            - Time information (np.ndarray)
            - Weights (np.ndarray)
            - Density values (np.ndarray)
            - Gradient of densities (np.ndarray)
        """
        return self.data
    
class PopulationEvalDataset(Dataset):
    potential: str = 'none'
    internal: str = 'none'
    beta: float = 0.0
    interaction: str = 'none'
    dt: float = 1.0
    T: int = 0
    data_dim: int = 0

    def __init__(self, key, dataset_name: str, solver: str, label='test_data'):
        """
        PopulationEvalDataset loads and organizes population trajectory data for evaluation.

        Parameters:
        key (Any): A key used for random number generation or seeding.
        dataset_name (str): The name of the dataset to load. Data should be located in 'data/{dataset_name}'.
        solver (str): The solver method used. Primarily for plotting purposes.
        label (str): Specifies whether to load 'test_data' or 'train_data'. Default is 'test_data'.
        """
        self.key = key
        self.solver = solver
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
        try:
            with open(os.path.join('data', dataset_name, 'args.txt'), 'r') as file:
                for line in file:
                    if "potential" in line:
                        self.potential = line.split("=")[1][:-1]
                    elif "internal" in line:
                        self.internal = line.split("=")[1][:-1]
                    elif "beta" in line:
                        self.beta = float(line.split("=")[1][:-1])
                    elif "interaction" in line:
                        self.interaction = line.split("=")[1][:-1]
                    elif "dt" in line:
                        self.dt = float(line.split("=")[1][:-1])
            self.trajectory_only_potential = self._compute_separate_predictions(
                potentials_all[self.potential] if self.potential != 'none' else False,
                False,
                False)
            self.trajectory_only_interaction = self._compute_separate_predictions(
                False,
                False,
                interactions_all[self.interaction] if self.interaction != 'none' else False)
        except FileNotFoundError:
            print(f"Dataset {dataset_name} does not have a ground truth file. Skipping error computation.")
            self.no_ground_truth = True

    def _compute_separate_predictions(
            self,
            potential: Callable[[jnp.ndarray], float],
            beta: float,
            interaction: Callable[[jnp.ndarray], float]
    ) -> jnp.ndarray:
        """
        Compute separate predictions based on potential, beta, and interaction.

        Parameters:
        potential (Any): Potential function or data used in predictions.
        beta (float): Beta parameter used in predictions.
        interaction (Any): Interaction function or data used in predictions.

        Returns:
        np.ndarray: Predictions for the population trajectories.
        """
        return get_SDE_predictions(
                    self.solver,
                    self.dt,
                    self.T,
                    1,
                    potential,
                    beta,
                    interaction,
                    self.key,
                    self.trajectory[0])


    def __len__(self) -> int:
        """
        Returns the number of particles at the first timestep.

        Returns:
        int: The number of particles in the first timestep.
        """
        return self.trajectory[0].shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieves a particle's features at the first timestep.

        Parameters:
        idx (int): The index of the particle to retrieve.

        Returns:
        np.ndarray: The features of the particle at the specified index and first timestep.
        """
        return self.trajectory[0][idx, :]

    def error_wasserstein(self, trajectory_predicted: np.ndarray) -> float:
        """
        Computes the Wasserstein loss between the predicted and true trajectories.

        Parameters:
        trajectory_predicted (np.ndarray): The predicted trajectory with shape (T, n_particles, n_features).

        Returns:
        float: The total Wasserstein error over all timesteps.
        """
        error = 0
        for t in range(1, trajectory_predicted.shape[0]):
            error += wasserstein_loss(
                        trajectory_predicted[t], jnp.asarray(self.trajectory[t]))
        return error
    
    def error_potential(self, trajectory_predicted: np.ndarray) -> float:
        """
        Computes the mean squared error between the predicted trajectory and the ground truth trajectory
        predicted using only the potential.

        Parameters:
        trajectory_predicted (np.ndarray): The predicted trajectory with shape (T, n_particles, n_features).

        Returns:
        float: The mean squared error considering only the potential.
        """
        return np.mean(np.sum((trajectory_predicted - self.trajectory_only_potential) ** 2, axis=(0, 2)))
    
    def error_internal(self, beta_predicted: float) -> float:
        """
        Computes the error in the internal parameter (beta).

        Parameters:
        beta_predicted (float): The predicted beta value.

        Returns:
        float: The error in the internal parameter, scaled by the trajectory length and time step.
        """
        return np.sqrt(2) * np.abs(np.abs(beta_predicted) - np.abs(self.beta)) * self.T * self.dt
    
    def error_interaction(self, trajectory_predicted: np.ndarray) -> float:
        """
        Computes the mean squared error between the predicted trajectory and the ground truth trajectory
        predicted using only the interaction.

        Parameters:
        trajectory_predicted (np.ndarray): The predicted trajectory with shape (T, n_particles, n_features).

        Returns:
        float: The mean squared error considering only the interaction.
        """
        return np.mean(np.sum((trajectory_predicted - self.trajectory_only_interaction) ** 2, axis=(0, 2)))

    def error_wasserstein_one_step_ahead(
            self,
            potential: Callable[[jnp.ndarray], float],
            beta: float,
            interaction: Callable[[jnp.ndarray], float],
            key_eval: jnp.ndarray,
            model: str,
            plot_folder_name: str
    ) -> jnp.ndarray:
        """
        Computes the Wasserstein error for one-step-ahead predictions.

        Parameters:
        potential (Callable[[jnp.ndarray], float]): Function for potential calculations, taking a JAX array and returning a float.
        beta (float): Beta parameter for predictions.
        interaction (Callable[[jnp.ndarray], float]): Function for interaction calculations, taking a JAX array and returning a float.
        key_eval (jnp.ndarray): Key used for random number generation.
        model (str): Solver used. For plotting (if applicable).
        plot_folder_name (str): Directory where plots should be saved.

        Returns:
        jnp.ndarray: Array of Wasserstein errors for one-step-ahead predictions.
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
                wasserstein_loss(predictions[-1], jnp.asarray(self.trajectory[t + 1])))
        return error_wasserstein_one_ahead

    def error_wasserstein_cumulative(
        self,
        predictions: jnp.ndarray,
        model: str,
        plot_folder_name: Optional[str] = None
    ) -> jnp.ndarray:
        """
        Computes the cumulative Wasserstein error per timestep.

        Parameters:
        predictions (jnp.ndarray): Array of predicted trajectories with shape (T+1, n_particles, n_features).
        model (str): Model object used for plotting (if applicable).
        plot_folder_name (Optional[str]): Directory where plots should be saved. If None, no plots are saved.

        Returns:
        jnp.ndarray: Array of cumulative Wasserstein errors.
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
                wasserstein_loss(predictions[t], jnp.asarray(self.trajectory[t])))
        return error_wasserstein_cumulative