"""
Module that implements the JKOnet* model based on the base interface.

The models are implemented using JAX and the FLAX library, following a functional paradigm to 
support efficient differentiation and optimization. The core classes include:

- ``JKOnetStar``: The full JKOnet* method, used for learning all the energy terms.
- ``JKOnetStarPotentialInternal``: A variant focusing on potential and internal energies.
- ``JKOnetStarPotential``: A variant focusing solely on the potential energy term.
- ``JKOnetStarTimePotential``: A time-extended variant of ``JKOnetStarPotential``.
- ``JKOnetStarLinear``: A model using linear parametrizations with various feature functions.
"""


import jax
import itertools
import functools
import jax.numpy as jnp
from flax.training import train_state
from utils.features import rbfs
from models.base import LearningDiffusionModel
from dataset import CouplingsDataset, LinearParametrizationDataset
from networks.energies import MLP
from networks.optim import get_optimizer, create_train_state, create_train_state_from_params
from networks.utils import network_grad, network_grad_time
from typing import Tuple, Callable, Union, Any, Dict
from flax.core import FrozenDict

class JKOnetStar(LearningDiffusionModel):
    """
    The full JKOnet* model for learning all energy terms.
    """
    def __init__(self, config: dict, data_dim: int, tau: float) -> None:
        """
        Initialize the JKOnetStar model.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing model and optimizer settings.
        data_dim : int
            Dimension of the input data.
        tau : float
            Represents the time scale over which the diffusion process described by the
            Fokker-Planck equation is considered.
        """
        super().__init__()
        self.tau = tau
        self.data_dim = data_dim

        # potential and interaction energies are vanilla MLPs
        self.layers = config['energy']['model']['layers']
        self.config_optimizer = config['energy']['optim']
        
        # create energy models
        self.model_potential = MLP(self.layers)
        self.model_internal = MLP([1])
        self.model_interaction = MLP(self.layers)
        
    def create_state(self, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
        """
        Create initial training states for the potential, internal, and interaction models.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random key for initialization.

        Returns
        -------
        Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            Tuple containing the training states for the potential, internal, and interaction models.
        """
        # to allow for jit compilation
        # train states
        potential = create_train_state(
            rng, self.model_potential, get_optimizer(self.config_optimizer), self.data_dim)
        internal = create_train_state(
            rng, self.model_internal, get_optimizer(self.config_optimizer), 1)
        interaction = create_train_state(
            rng, self.model_interaction, get_optimizer(self.config_optimizer), self.data_dim)
        return potential, internal, interaction

    def create_state_from_params(
        self,
        potential_params: dict,
        internal_params: dict,
        interaction_params: dict
    ) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
        """
        Create training states from the provided parameters.

        Parameters
        ----------
        potential_params : dict
            Parameters for the potential model.
        internal_params : dict
            Parameters for the internal model.
        interaction_params : dict
            Parameters for the interaction model.

        Returns
        -------
        Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            Tuple containing the training states for the potential, internal, and interaction models.
        """
        potential = create_train_state_from_params(
            self.model_potential, potential_params, get_optimizer(self.config_optimizer))
        internal = create_train_state_from_params(
            self.model_internal, internal_params, get_optimizer(self.config_optimizer))
        interaction = create_train_state_from_params(
            self.model_interaction, interaction_params, get_optimizer(self.config_optimizer))
        return potential, internal, interaction

    def get_params(self, state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]) -> Tuple[FrozenDict[str, Any], FrozenDict[str, Any], FrozenDict[str, Any]]:
        """
        Get parameters from the training state.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            Training state containing potential, internal, and interaction models.

        Returns
        -------
        Tuple[dict, dict, dict]
            Tuple containing the parameters for the potential, internal, and interaction models.
        """
        potential_state, internal_state, interaction_state = state
        return potential_state.params, internal_state.params, interaction_state.params
        
    def _loss_potential_term(
        self,
        potential_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the loss term associated with the potential model.

        Parameters
        ----------
        potential_params : dict
            Parameters for the potential model.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values.
        rho_grad : jnp.ndarray
            Gradient of density values.

        Returns
        -------
        jnp.ndarray
            Loss term for the potential model.
        """
        # need potential_state as parameter to compute the gradient
        return network_grad(self.model_potential, potential_params)(ys)
    
    def _loss_internal_term(
        self,
        internal_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the loss term associated with the internal model.

        Parameters
        ----------
        internal_params : dict
            Parameters for the internal model.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values.
        rho_grad : jnp.ndarray
            Gradient of density values.

        Returns
        -------
        jnp.ndarray
            Loss term for the internal model.
        """
        # need internal_state as parameter to compute the gradient
        beta = self.model_internal.apply({'params': internal_params}, jnp.asarray([1]))
        return beta * rho_grad / rho[:, None]
    
    def _loss_interaction_term(
        self,
        interaction_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the loss term associated with the interaction model.

        Parameters
        ----------
        interaction_params : dict
            Parameters for the interaction model.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values.
        rho_grad : jnp.ndarray
            Gradient of density values.

        Returns
        -------
        jnp.ndarray
            Loss term for the interaction model.
        """
        # need interaction_state as parameter to compute the gradient
        interaction_grad = network_grad(self.model_interaction,
                                        interaction_params)
        def loss_energy_interaction(p):
                return jnp.mean(interaction_grad(p - ys), axis=0)
        return jax.vmap(loss_energy_interaction)(ys)

    def loss(
        self,
        potential_params: dict,
        internal_params: dict,
        interaction_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the total loss for the model by combining potential, internal, and interaction terms.

        Parameters
        ----------
        potential_params : dict
            Parameters for the potential model.
        internal_params : dict
            Parameters for the internal model.
        interaction_params : dict
            Parameters for the interaction model.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values.
        rho_grad : jnp.ndarray
            Gradient of density values.

        Returns
        -------
        jnp.ndarray
            Total loss value.
        """
        # need all states as parameters to compute the gradients
        return jnp.sum(ws * jnp.sum((self.tau * (
                self._loss_potential_term(
                    potential_params, xs, ys, ws, rho, rho_grad) + \
                self._loss_internal_term(
                    internal_params, xs, ys, ws, rho, rho_grad) + \
                self._loss_interaction_term(
                    interaction_params, xs, ys, ws, rho, rho_grad)
            ) + (ys - xs)) ** 2, axis=1))
        
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
        potential, _, _ = state
        return lambda x: potential.apply_fn({'params': potential.params}, x)
    
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
        _, _, interaction = state
        return lambda x: interaction.apply_fn({'params': interaction.params}, x)
    
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
        _, internal, _ = state
        return abs(internal.apply_fn({'params': internal.params}, jnp.asarray([1])).item())

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
        xs, ys, t, ws, rho, rho_grad = sample
        return self._train_step(state, xs, ys, t, ws, rho, rho_grad)

    def _train_step(
        self,
        state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        t: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:
        """
        Execute a training step by calculating gradients and updating model parameters.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            Training state containing potential, internal, and interaction models.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        t : jnp.ndarray
            Timestep of the target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values.
        rho_grad : jnp.ndarray
            Gradient of density values.

        Returns
        -------
        Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]
            The loss value and the updated training states.
        """
        potential, internal, interaction = state
        loss, grads = jax.value_and_grad(
                self.loss, argnums=(0, 1, 2))(
                    potential.params, 
                    internal.params,
                    interaction.params,
                    xs, ys, ws, rho, rho_grad)
        potential = potential.apply_gradients(grads=grads[0])
        internal = internal.apply_gradients(grads=grads[1])
        interaction = interaction.apply_gradients(grads=grads[2])

        return loss, (potential, internal, interaction)

    def load_dataset(self, dataset_name: str) -> CouplingsDataset:
        """
        Load and return a dataset based on the given dataset name.

        This method creates an instance of the `CouplingsDataset` class using the specified dataset name.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load. This name is used to locate and initialize the dataset.

        Returns
        -------
        CouplingsDataset
            An instance of the `CouplingsDataset` class, which contains the loaded dataset.
        """
        return CouplingsDataset(dataset_name)

    
class JKOnetStarPotentialInternal(JKOnetStar):
    """
    A specialized variant of the JKOnetStar model that only considers potential and internal terms.
    """
    def loss(
        self,
        potential_params: dict,
        internal_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the loss for the potential and internal terms of the model.

        Parameters
        ----------
        potential_params : dict
            Parameters of the potential model.
        internal_params : dict
            Parameters of the internal model.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values for the data samples.
        rho_grad : jnp.ndarray
            Gradient of the density values.

        Returns
        -------
        jnp.ndarray
            The computed loss value.
        """
        # need potential_state and internal_state as parameters to compute the gradients
        return jnp.sum(ws * jnp.sum((self.tau * (
                self._loss_potential_term(
                    potential_params, xs, ys, ws, rho, rho_grad) + \
                self._loss_internal_term(
                    internal_params, xs, ys, ws, rho, rho_grad)
            ) + (ys - xs)) ** 2, axis=1))
    
    def get_interaction(self, _) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Returns a function representing the interaction term.

        This implementation returns a constant zero function, as the interaction is not used in this variant.

        Parameters
        ----------
        _ : Any
            Unused parameter in this context.

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            A function that always returns 0.
        """
        return lambda _: 0.

    def _train_step(
        self,
        state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        t: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:
        """
        Perform a single training step.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            The current states of the potential, internal, and interaction models.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        t : jnp.ndarray
            Timestep of the target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values for the data samples.
        rho_grad : jnp.ndarray
            Gradient of the density values.

        Returns
        -------
        Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]
            The loss value and the updated states of the potential and internal models.
        """
        potential, internal, _ = state
        loss, grads = jax.value_and_grad(
                self.loss, argnums=(0, 1))(
                    potential.params, internal.params, xs, ys, ws, rho, rho_grad)
        potential = potential.apply_gradients(grads=grads[0])
        internal = internal.apply_gradients(grads=grads[1])

        return loss, (potential, internal, _)
    

class JKOnetStarPotential(JKOnetStarPotentialInternal):
    """
    A variant of the JKOnetStar model to learn only the potential term.
    """
    def loss(
        self,
        potential_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the loss for the potential term of the model.

        Parameters
        ----------
        potential_params : dict
            Parameters of the potential model.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values for the data samples.
        rho_grad : jnp.ndarray
            Gradient of the density values.

        Returns
        -------
        jnp.ndarray
            The computed loss value.
        """
        # need potential_state as parameter to compute the gradient
        return jnp.sum(ws * jnp.sum((self.tau * (
                self._loss_potential_term(potential_params, xs, ys, ws, rho, rho_grad)
            ) + (ys - xs)) ** 2, axis=1))
    
    def _train_step(
        self,
        state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        t: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:
        """
        Perform a single training step.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            The current states of the potential, internal, and interaction models.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        t : jnp.ndarray
            Timestep of the target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values for the data samples.
        rho_grad : jnp.ndarray
            Gradient of the density values.

        Returns
        -------
        Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]
            The loss value and the updated states of the potential model, with the internal and interaction models unchanged.
        """
        potential, _, _ = state
        loss, grads = jax.value_and_grad(
                self.loss, argnums=0)(
                    potential.params, xs, ys, ws, rho, rho_grad)
        potential = potential.apply_gradients(grads=grads)

        return loss, (potential, _, _)
    
    
    def get_beta(self, state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]) -> float:
        """
        Return a constant zero value for the beta term of the internal energy model.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            The current states of the potential, internal, and interaction models.

        Returns
        -------
        float
            The constant zero value for the beta term of the internal energy model.
        """
        return 0.

class JKOnetStarTimePotential(JKOnetStarPotential):
    """
    A variant of the JKOnetStarPotential model that incorporates time information in the potential term.
    """
    def create_state(self, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
        """
        Creates initial training states for the potential, internal, and interaction models.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random key for JAX-based random number generation.

        Returns
        -------
        Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            The initial states for the potential, internal, and interaction models.
        """
        # to allow for jit compilation
        # train states
        potential = create_train_state(
            rng, self.model_potential, get_optimizer(self.config_optimizer), self.data_dim + 1)
        internal = create_train_state(
            rng, self.model_internal, get_optimizer(self.config_optimizer), 1)
        interaction = create_train_state(
            rng, self.model_interaction, get_optimizer(self.config_optimizer), self.data_dim)
        return potential, internal, interaction

    def _loss_potential_term(
        self,
        potential_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        t: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the loss contribution from the potential term, including time information.

        Parameters
        ----------
        potential_params : dict
            Parameters of the potential model.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        t : jnp.ndarray
            Timestep of the target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values for the data samples.
        rho_grad : jnp.ndarray
            Gradient of the density values.

        Returns
        -------
        jnp.ndarray
            The computed potential loss contribution.
        """
        ys_concat = jnp.concatenate((ys, t[:, None]), axis=1)
        return network_grad_time(self.model_potential, potential_params)(ys_concat)
    def loss(
        self,
        potential_params: dict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        t: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the total loss for the model, considering the potential term with time information.

        Parameters
        ----------
        potential_params : dict
            Parameters of the potential model.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        t : jnp.ndarray
            Timestep of the target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values for the data samples.
        rho_grad : jnp.ndarray
            Gradient of the density values.

        Returns
        -------
        jnp.ndarray
            The computed loss value.
        """
        return jnp.sum(ws * jnp.sum((self.tau * (
            self._loss_potential_term(potential_params, xs, ys, t, ws, rho, rho_grad)
        ) + (ys - xs)) ** 2, axis=1))

    def _train_step(
        self,
        state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        t: jnp.ndarray,
        ws: jnp.ndarray,
        rho: jnp.ndarray,
        rho_grad: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:
        """
        Performs a single training step by updating the potential model.

        Parameters
        ----------
        state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
            The current states of the potential, internal, and interaction models.
        xs : jnp.ndarray
            Initial particle distribution.
        ys : jnp.ndarray
            Target particle distribution.
        t : jnp.ndarray
            Timestep of the target particle distribution.
        ws : jnp.ndarray
            Weights of the couplings.
        rho : jnp.ndarray
            Density values for the data samples.
        rho_grad : jnp.ndarray
            Gradient of the density values.

        Returns
        -------
        Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]
            The loss value and the updated states of the potential model, with the internal and interaction models unchanged.
        """
        potential, _, _ = state
        loss, grads = jax.value_and_grad(
            self.loss, argnums=0)(
            potential.params, xs, ys, t, ws, rho, rho_grad)
        potential = potential.apply_gradients(grads=grads)

        return loss, (potential, _, _)


class JKOnetStarLinear(LearningDiffusionModel):
    """
    The linear parametrization of the JKOnet* model.
    """

    def __init__(
            self,
            config: Dict[str, Union[Dict[str, Union[int, float, bool]], float]],
            data_dim: int,
            tau: float
    ) -> None:
        """
        Initializes the JKOnetStarLinear model with configuration and data dimensions.

        Parameters
        ----------
        config : dict
            Configuration dictionary specifying model parameters and feature settings.
        data_dim : int
            Dimensionality of the input data.
        tau : float
            Represents the time scale over which the diffusion process described by the
            Fokker-Planck equation is considered.
        """
        super().__init__()
        self.tau = tau
        self.data_dim = data_dim
        self.config_features = config['energy']['linear']['features']
        self.reg = config['energy']['linear']['reg']

        self.fns = []
        if 'polynomials' in self.config_features:
            exps = [jnp.asarray(e)
                for e in itertools.product(range(
                self.config_features['polynomials']['degree'] + 1), 
                repeat=self.data_dim)
                if sum(e) > 0]
            self.fns += [
                functools.partial(
                    lambda v, e: jnp.prod(v ** e), e=e)
                for e in exps]
            if self.config_features['polynomials']['sines']:
                self.fns += [
                    functools.partial(
                        lambda v, e: jnp.prod(jnp.sin(v ** e)), e=e)
                    for e in exps]
                self.fns += [
                    functools.partial(
                        lambda v, e: jnp.prod(jnp.sin(v) ** e), e=e)
                    for e in exps if max(e) < 3]
            if self.config_features['polynomials']['cosines']:
                self.fns += [
                    functools.partial(
                        lambda v, e: jnp.prod(jnp.cos(v ** e)), e=e)
                    for e in exps]
                self.fns += [
                    functools.partial(
                        lambda v, e: jnp.prod(jnp.cos(v) ** e), e=e)
                    for e in exps if max(e) < 3]
                
        if 'rbfs' in self.config_features:
            domain = self.config_features['rbfs']['domain']
            n_centers = self.config_features['rbfs']['n_centers_per_dim']
            sigma = self.config_features['rbfs']['sigma']
            centers = [jnp.asarray(c) 
                       for c in itertools.product(
                           jnp.linspace(domain[0], domain[1], n_centers),repeat=self.data_dim)]
            self.fns += [
                functools.partial(
                    lambda x, c, t: jnp.exp(-jnp.sum((x - c) ** 2) / sigma) * t(x, c), c=c, t=rbfs[type])
                for type in self.config_features['rbfs']['types']
                for c in centers
            ]

        # compute yts
        _features_grad = jax.vmap(self.features_grad)
        self.yt1 = lambda xs: _features_grad(xs)
        self.yt2 = lambda xs: jax.vmap(lambda x: jnp.mean(_features_grad(x[None, :] - xs), axis=0))(xs)
        self.yt3 = lambda rho, rho_grad: (rho_grad / rho[:, None])[:, :, None]
        self.theta_dim = self.features_dim * 2 + 1
        self.unpack_theta1 = lambda theta: theta[:self.features_dim]
        self.unpack_theta2 = lambda theta: theta[self.features_dim:-1]
        self.unpack_theta3 = lambda theta: theta[-1]

        no_feature_fn = lambda xs: jnp.zeros((xs.shape[0], self.data_dim, 0))
        if not config['energy']['linear']['potential']:
            self.yt1 = no_feature_fn
            self.theta_dim -= self.features_dim
            self.unpack_theta1 = lambda _: jnp.zeros((self.features_dim, 1))
            self.unpack_theta2 = lambda theta: theta[:self.features_dim]

        if not config['energy']['linear']['internal']:
            self.yt3 = lambda rho, _: jnp.zeros((rho.shape[0], self.data_dim, 0))
            self.theta_dim -= 1
            self.unpack_theta3 = lambda _: jnp.zeros((1,))

        if not config['energy']['linear']['interaction']:
            self.yt2 = no_feature_fn
            self.theta_dim -= self.features_dim
            self.unpack_theta2 = lambda _: jnp.zeros((self.features_dim, 1))



    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the feature functions for the input data `x`.

        Parameters
        ----------
        x : jnp.ndarray
            Input data for which to compute feature functions.

        Returns
        -------
        jnp.ndarray
            The computed feature functions for the input data.
        """
        return jnp.asarray([f(x) for f in self.fns])

    def features_grad(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradients of the feature functions with respect to `x`.

        Parameters
        ----------
        x : jnp.ndarray
            Input data for which to compute gradients of feature functions.

        Returns
        -------
        jnp.ndarray
            The gradients of the feature functions with respect to the input data.
        """
        return jnp.stack([jax.grad(f)(x) for f in self.fns], axis=1)
    
    @property
    def features_dim(self) -> int:
        """
        The dimension of the feature space.

        Computes and caches the dimension of the feature space based on the feature functions.

        Returns
        -------
        int
            The dimension of the feature space.
        """
        if not hasattr(self, '_features_dim_cache'):
            self._features_dim_cache = self.features(jnp.ones((self.data_dim,))).shape[0]
        return self._features_dim_cache
        
    def create_state(self, _) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Creates the initial state for the model.

        This method returns a tuple of zero-initialized arrays corresponding to the potential, interaction, and internal parameters.

        Parameters
        ----------
        _ : Ignored
            Placeholder for compatibility, not used.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            The initial state consisting of zero-initialized potential, interaction, and internal parameters.
        """
        return (
            jnp.zeros((self.features_dim, 1)), 
            jnp.zeros((self.features_dim, 1)), 
            jnp.zeros((1, 1))
        )
    
    def create_state_from_params(
        self,
        potential_params: jnp.ndarray,
        interaction_params: jnp.ndarray,
        internal_params: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Creates the state from given parameters.

        Parameters
        ----------
        potential_params : jnp.ndarray
            The parameters for the potential term.
        interaction_params : jnp.ndarray
            The parameters for the interaction term.
        internal_params : jnp.ndarray
            The parameters for the internal term.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            A tuple containing the potential, interaction, and internal parameters.
        """
        return(
            potential_params,
            interaction_params,
            internal_params,
        )
    
    def get_params(self, state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Retrieves the parameters from the given state.

        Parameters
        ----------
        state : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            The current state containing the potential, interaction, and internal parameters.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            The potential, interaction, and internal parameters extracted from the state.
        """
        potential_params, internal_params, interaction_params = state
        return potential_params, internal_params, interaction_params


    def train_step(self, _, all_samples: Tuple) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Performs a single training step, updating the model's parameters.

        This method solves the least squares problem to update the model parameters based on the provided samples.

        Parameters
        ----------
        _ : Ignored
            Placeholder for compatibility, not used.
        all_samples : Tuple
            A tuple of all the samples, where each sample contains (xs, ys, t, ws, rho, rho_grad).

        Returns
        -------
        Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
            The error and the updated parameters (potential, interaction, internal).
        """
        A = jnp.eye(self.theta_dim) * self.reg
        b = jnp.zeros((self.theta_dim,))
            
        for xs, ys, _, ws, rho, rho_grad in all_samples:
            # unbatch
            xs = xs.squeeze(axis=0)
            ys = ys.squeeze(axis=0)
            ws = ws.squeeze(axis=0)
            rho = rho.squeeze(axis=0)
            rho_grad = rho_grad.squeeze(axis=0)

            yt = jnp.concatenate([
                self.yt1(xs), 
                self.yt2(xs), 
                self.yt3(rho, rho_grad)], 
                axis=2)

            A += jnp.mean(jnp.einsum('ijk,ijh->ikh', yt, yt), axis=0)
            b += jnp.sum(
                ws[:, None] * jnp.einsum('ijk,ij->ik', yt, ys - xs), axis=0)

        sol = jnp.linalg.solve(A, b)
        theta = - sol / self.tau
        err = jnp.sum((A @ sol - b) ** 2)

        return err, (
            self.unpack_theta1(theta),
            self.unpack_theta2(theta),
            self.unpack_theta3(theta)
        )
    
    def get_potential(self, state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Returns the potential function based on the current state.

        Parameters
        ----------
        state : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            The current state containing the potential, interaction, and internal parameters.

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            A function that computes the potential for a given input `x`.
        """
        theta1, _, _ = state
        return lambda x: jnp.sum(theta1 * self.features(x))
    
    def get_interaction(self, state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Returns the interaction function based on the current state.

        Parameters
        ----------
        state : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            The current state containing the potential, interaction, and internal parameters.

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            A function that computes the interaction for a given input `x`.
        """
        _, theta2, _ = state
        return lambda x: jnp.sum(theta2 * self.features(x))
    
    def get_beta(self, state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> float:
        """
        Returns the value of the internal parameter (beta) based on the current state.

        Parameters
        ----------
        state : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            The current state containing the potential, interaction, and internal parameters.

        Returns
        -------
        float
            The beta value from the internal energy model.
        """
        _, _, theta3 = state
        return jnp.abs(theta3).item()

    def load_dataset(self, dataset_name: str) -> LinearParametrizationDataset:
        """
        Loads and returns the dataset for linear parametrizations.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load.

        Returns
        -------
        LinearParametrizationDataset
            The dataset object for linear parametrizations.
        """
        return LinearParametrizationDataset(dataset_name)
