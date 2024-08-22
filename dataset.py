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

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.time[idx], self.weight[idx], self.densities[idx], self.densities_grads[
            idx]
    
class LinearParametrizationDataset(Dataset):
    def __init__(self, dataset_name: str):
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
        
    def __len__(self):
        return 1
    
    def __getitem__(self, _):
        return self.data
    
class PopulationEvalDataset(Dataset):
    potential = 'none'
    internal = 'none'
    beta = 0.0
    interaction = 'none'
    dt = 1
    T = 0
    data_dim = 0

    def __init__(self, key, dataset_name: str, solver:str, label='test_data'):
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
        # self.dt = unique_labels[1]-unique_labels[0]
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
    def _compute_separate_predictions(self, potential, beta, interaction):
        return get_SDE_predictions(
                    self.solver,
                    self.dt,
                    self.T,
                    1,
                    False,
                    False,
                    interaction,
                    self.key,
                    self.trajectory[0])


    def __len__(self):
        return self.trajectory[0].shape[0]

    def __getitem__(self, idx):
        # returns a particle at the first timestep
        # batching means getting more particles per timestep
        return self.trajectory[0][idx, :]

    def error_wasserstein(self, trajectory_predicted):
        error = 0
        for t in range(1, trajectory_predicted.shape[0]):
            error += wasserstein_loss(
                        trajectory_predicted[t], jnp.asarray(self.trajectory[t]))
        return error
    
    def error_potential(self, trajectory_predicted):
        return np.mean(np.sum((trajectory_predicted - self.trajectory_only_potential) ** 2, axis=(0, 2)))
    
    def error_internal(self, beta_predicted):
        return np.sqrt(2) * np.abs(np.abs(beta_predicted) - np.abs(self.beta)) * self.T * self.dt
    
    def error_interaction(self, trajectory_predicted):
        return np.mean(np.sum((trajectory_predicted - self.trajectory_only_interaction) ** 2, axis=(0, 2)))

    def error_wasserstein_one_step_ahead(self, potential, beta, interaction, key_eval, model, plot_folder_name=None):
        error_wasserstein_one_ahead = jnp.ones(self.T)
        for t in range(self.T):
            init = self.trajectory[t]
            timepoint_ahead = get_SDE_predictions(
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
                    timepoint_ahead[-1].reshape(1, -1, self.data_dim),
                    self.trajectory,
                    interval=(t + 1, t + 1),
                    model=model,
                    save_to=plot_path)
                plt.close(prediction_fig)
            error_wasserstein_one_ahead = error_wasserstein_one_ahead.at[t].set(
                wasserstein_loss(timepoint_ahead[-1], jnp.asarray(self.trajectory[t + 1])))
        return error_wasserstein_one_ahead

    def error_wasserstein_cumulative(self, predictions, model, plot_folder_name=None):
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