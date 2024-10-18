"""
Gaussian Mixture Model (GMM) Module for Trajectory Data

This module defines the ``GaussianMixtureModel`` class, which fits a Gaussian Mixture Model (GMM) to trajectory data and provides methods to compute density estimates. The class also allows saving and loading the model parameters to/from a file.

The GMM is fitted to time-dependent trajectory data, where each timestep contains a set of data points, and the class allows for computing the GMM-based density at any given timestep for a new input point.

Dependencies:
-------------
- ``jax.numpy``: Used for array manipulation and numerical computations.
- ``sklearn.mixture.GaussianMixture``: Used to fit the Gaussian Mixture Model to data.
- ``pickle``: Used for saving and loading model parameters.
- ``chex``: Provides utilities for type and dimensionality checks.
- ``typing.List``, ``typing.Dict``: Used for type hinting.

Class:
------
- ``GaussianMixtureModel``: Handles the fitting of GMMs to trajectory data, saving/loading the model, and computing the density for a given input.

Class Attributes:
-----------------
- ``gms_means``: A list storing the means of the Gaussian components for each timestep.
- ``gms_covs_invs``: A list storing the inverses of the covariance matrices for each timestep.
- ``gms_den``: A list of normalization factors for density computation at each timestep.
- ``gms_weights``: A list storing the weights of each Gaussian component for each timestep.

Methods:
--------
- ``__init__``: Initializes the class with empty lists to hold model parameters.
- ``fit``: Fits a GMM to the given trajectory data and stores the relevant parameters (means, inverse covariances, weights).
- ``to_file``: Saves the model parameters to a file using `pickle`.
- ``from_file``: Loads the model parameters from a file using `pickle`.
- ``gmm_density``: Computes the density of the GMM at a specified timestep for a given data point.

Example Usage:
--------------
.. code-block:: python

    from module_name import GaussianMixtureModel
    import jax.numpy as jnp

    # Initialize the GMM model
    gmm = GaussianMixtureModel()

    # Example trajectory data (for multiple timesteps)
    trajectory = {
        0: jnp.array([[1.0, 2.0], [2.0, 3.0]]),
        1: jnp.array([[1.5, 2.5], [2.5, 3.5]])
    }

    # Fit the GMM to the trajectory data with 2 components
    gmm.fit(trajectory, n_components=2)

    # Compute the density at timestep 0 for a new data point
    x = jnp.array([1.2, 2.2])
    density = gmm.gmm_density(t=0, x=x)

    # Save the model to a file
    gmm.to_file("gmm_model.pkl")

    # Load the model from a file
    gmm.from_file("gmm_model.pkl")
"""

import jax.numpy as jnp
from sklearn.mixture import GaussianMixture
import pickle
import chex
from typing import List, Dict

class GaussianMixtureModel:
    """
    A class to represent a Gaussian Mixture Model (GMM) that can be fitted to trajectory data and allows for density computation.
    
    Attributes
    ----------
    gms_means : List[jnp.ndarray]
        List to store means of the Gaussian components for each timestep.
    gms_covs_invs : List[jnp.ndarray]
        List to store the inverses of the covariance matrices for each timestep.
    gms_den : List[float]
        List to store normalization factors (density denominators) for each timestep.
    gms_weights : List[float]
        List to store the weights of each Gaussian component for each timestep.
    """
    
    def __init__(self):
        """
        Initializes the GaussianMixtureModel class.
        """
        self.gms_means: List[jnp.ndarray] = []
        self.gms_covs_invs: List[jnp.ndarray] = []
        self.gms_den: List[float] = []
        self.gms_weights: List[float] = []

    def fit(self, trajectory: dict, n_components: int, seed: int) -> None:
        """
        Fits a Gaussian Mixture Model (GMM) to the given trajectory data.

        Parameters
        ----------
        trajectory : dict
            A dictionary where each key is a timestep and each value is a 2D array (n_samples, n_features) of data points.
        n_components : int
            The number of clusters (components) to use in the GMM.
        seed : int
            Random seed for reproducibility.
        """
        for _, val in trajectory.items():
            chex.assert_type(val, float)
            chex.assert_rank(val, 2)  # Check that each value in trajectory is a 2D array

        data_dim = list(trajectory.values())[0].shape[1]
        for label in sorted(trajectory.keys()):
            data = trajectory[label]
            gm = GaussianMixture(n_components=n_components, random_state=seed)
            gm.fit(data)

            # Discard components with small determinants
            covariances = gm.covariances_
            dets = jnp.asarray([jnp.linalg.det(covariances[i]) for i in range(n_components)])
            idxs = jnp.where(jnp.greater(dets, 1e-4))

            # Store density parameters
            self.gms_means.append(gm.means_[idxs])
            self.gms_covs_invs.append(jnp.linalg.inv(gm.covariances_[idxs]))
            self.gms_den.append(1 / jnp.sqrt((2 * jnp.pi) ** data_dim * dets[idxs]))
            self.gms_weights.append(jnp.asarray(gm.weights_[idxs] / jnp.sum(gm.weights_[idxs])))

    def to_file(self, 
                filename: str):
        """
        Saves the GMM model parameters to a file.

        Parameters
        ----------
        filename : str
            The file path to save the model to.
        """
        data = {
            'gms_means': self.gms_means,
            'gms_covs_invs': self.gms_covs_invs,
            'gms_den': self.gms_den,
            'gms_weights': self.gms_weights
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def from_file(self, 
                  filename: str):
        """
        Loads the GMM model parameters from a file.

        Parameters
        ----------
        filename : str
            The file path to load the model from.
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.gms_means = data['gms_means']
            self.gms_covs_invs = data['gms_covs_invs']
            self.gms_den = data['gms_den']
            self.gms_weights = data['gms_weights']

    def gmm_density(self, t: int, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the GMM density for a given timestep and data point.

        Parameters
        ----------
        t : int
            The timestep to use for computing the GMM density.
        x : jnp.ndarray
            The data point (array of shape (n_features,)) for which to calculate the density.

        Returns
        -------
        jnp.ndarray
            The computed density value at the specified time and state.
        """
        diffs = x - self.gms_means[t]  # (n_components, dim)
        mahalanobis_terms = jnp.einsum('ij,ijk,ik->i', diffs, self.gms_covs_invs[t], diffs)  # (n_components,)
        exponent_terms = jnp.exp(-0.5 * mahalanobis_terms)  # (n_components,)
        weighted_terms = self.gms_weights[t] * self.gms_den[t] * exponent_terms  # (n_components,)
        result = jnp.sum(weighted_terms)  # Scalar value
        return jnp.clip(result, a_min=0.00001)