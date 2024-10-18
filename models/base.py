"""
Module for implementing and comparing different learning diffusion models using JAX.

This module provides an interface for writing code that seamlessly interface with the training and testing procedures.
"""

from abc import abstractmethod
from torch.utils.data import Dataset
import jax
import jax.numpy as jnp
from flax.training import train_state
from typing import Any, Callable, Tuple

class LearningDiffusionModel:
    """
    An abstract base class representing a learning diffusion model, designed for use with JAX.

    This interface serves as a blueprint for implementing new models to learn the different diffusion terms.
    """
    def __init__(self) -> None:
        pass

    def load_dataset(self, dataset_name: str) -> Dataset:
        """
        Loads and prepares the specified dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load.

        Returns
        -------
        Dataset
            The loaded dataset.
        """
        pass
    
    @abstractmethod
    def create_state(self, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
        """
        Abstract method to initialize and return the model's state. Must be implemented by subclasses.

        Parameters
        ----------
        rng : PRNGKey
            The random key to use for initializing the state.

        Returns
        -------
        Tuple[TrainState, TrainState, TrainState]
            A tuple containing the model's state for the potential, internal, and interaction terms.

            Note: You can use less than three states if your model does not require all three terms, or if it uses a single, shared, state.
        """
        pass

    @abstractmethod
    def train_step(
            self,
            state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
            sample: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:
        """
        Perform a single training step.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            Training state containing potential, internal, and interaction models.
        sample : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Training data sample consisting of xs, ys, t, ws, rho, and rho_grad.

        Returns
        -------
        Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]
            The loss value and the updated training states.
        """
        pass

    def get_potential(self, state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Get the potential function from the model state.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            Training state containing potential, internal, and interaction models.

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            Function that computes the potential.
        """
        return lambda x: 0.
    
    def get_beta(self, state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]) -> float:
        """
        Get the beta value from the model state.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            Training state containing potential, internal, and interaction models.

        Returns
        -------
        float
            The beta value from the internal energy model.
        """
        return 0.
    
    def get_interaction(self, state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Get the interaction function from the model state.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            Training state containing potential, internal, and interaction models.

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            Function that computes the interaction.
        """
        return lambda x: 0.