"""
This module contains the implementation of the JKOnet model and of the Monge gap regularization.

- For JKOnet see https://arxiv.org/abs/2106.06345
- For the Monge gap regularizer see https://arxiv.org/abs/2302.04953

Source: https://github.com/bunnech/jkonet
"""


import jax
import jax.numpy as jnp
import numpy as np
import optax
from models.base import LearningDiffusionModel
from dataset import PopulationDataset
from networks.energies import MLP
from networks.icnns import ICNN
from networks.optim import get_optimizer, create_train_state, penalize_weights_icnn, create_train_state_from_params
from networks.fixpoint_loop import fixpoint_iter
from networks.utils import count_parameters
from utils.ot import sinkhorn_loss
from ott.neural.methods.monge_gap import monge_gap_from_samples as monge_gap
from flax.training import train_state
from typing import Any, Dict, Tuple, List, Callable


class JKOnet(LearningDiffusionModel):
    """
    This code ports https://github.com/bunnech/jkonet to our interface.
    See https://arxiv.org/abs/2106.06345
    """
    def load_dataset(self, dataset_name: str) -> PopulationDataset:
        return PopulationDataset(dataset_name, self.batch_size)
    
    def __init__(self, config: Dict, data_dim: int, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.data_dim = data_dim
        self.batch_size = config['train']['batch_size']
        self.potential_optimizer = config['energy']['optim']
        self.model_potential = MLP(config['energy']['model']['layers'])


        # otmap
        self.config_settings = config['settings']
        self.otmap_config = config['otmap']
        self.otmap_optimizer = get_optimizer(config['otmap']['optim'])

    def _loss_fn_otmap(
        self,
        params_otmap: Any,
        params_energy: Any,
        data: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        grad_otmap_data = jax.vmap(lambda x: jax.grad(
            self.model_otmap.apply, argnums=1)(
                {'params': params_otmap}, x))(data)
        predicted = self.config_settings['cvx_reg'] * data + grad_otmap_data

        # jko objective
        loss_e = jnp.mean(jax.vmap(lambda v: self.model_potential.apply(
            {'params': params_energy}, v))(predicted))
        loss_p = jnp.mean(jnp.sum((predicted - data) ** 2, axis=1))
        loss = loss_e + 1 / (2 * self.tau) * loss_p

        # add penalty to negative icnn weights in relaxed setting
        if not self.otmap_config['model']['pos_weights']:
            penalty = penalize_weights_icnn(params_otmap)
            loss += self.otmap_config['optim']['beta'] * penalty

        return loss, predicted

    def _prepare_otmap(self) -> ICNN:
        return ICNN(dim_hidden=self.otmap_config['model']['layers'],
                    init_fn=self.otmap_config['model']['init_fn'],
                    pos_weights=self.otmap_config['model']['pos_weights'])

    def create_state(self, rng: jax.random.PRNGKey) -> Any:
        self.rng = rng
        self.model_otmap = self._prepare_otmap()
        self.optimize_otmap_fn = get_optimize_psi_fn(
            jax.jit(self._loss_fn_otmap),
            self.otmap_optimizer, 
            self.otmap_config['optim']['n_iter'],
            self.otmap_config['optim']['min_iter'], 
            self.otmap_config['optim']['max_iter'],
            self.otmap_config['optim']['inner_iter'], 
            self.otmap_config['optim']['thr'],
            self.config_settings['fploop'])
        potential = create_train_state(
            rng, self.model_potential, get_optimizer(self.potential_optimizer), self.data_dim)
        return potential
    
    def create_state_from_params(self, params: Dict) -> train_state.TrainState:
        self.model_otmap = self._prepare_otmap()
        self.optimize_otmap_fn = get_optimize_psi_fn(
            jax.jit(self._loss_fn_otmap),
            self.otmap_optimizer, 
            self.otmap_config['optim']['n_iter'],
            self.otmap_config['optim']['min_iter'], 
            self.otmap_config['optim']['max_iter'],
            self.otmap_config['optim']['inner_iter'], 
            self.otmap_config['optim']['thr'],
            self.config_settings['fploop'])
        potential = create_train_state_from_params(
            self.model_potential,  params, get_optimizer(self.potential_optimizer))
        return potential
        

    # Source: https://github.com/bunnech/jkonet
    def loss_fn_energy(
            self,
            params_energy: Dict,
            rng_psi: jax.random.PRNGKey,
            batch: jnp.ndarray,
            t: int
    ) -> Tuple[float, Tuple[jnp.ndarray, jnp.ndarray]]:
        # initialize psi model and optimizer
        params_psi = self.model_otmap.init(
            rng_psi, jnp.ones(batch[t].shape[1]))['params']
        opt_state_psi = self.otmap_optimizer.init(params_psi)

        # solve jko step
        _, predicted, loss_psi = self.optimize_otmap_fn(
            params_energy, params_psi, opt_state_psi, batch[t])

        # compute distance between prediction and data
        loss_energy = sinkhorn_loss(predicted, batch[t + 1], self.config_settings['epsilon'])

        return loss_energy, (loss_psi, predicted)
    
    def train_step(
        self,
        state: train_state.TrainState,
        sample: List[List[np.ndarray]]
    ) -> Tuple[jnp.ndarray, train_state.TrainState]:
        batch = jnp.stack(sample, axis=2).transpose(2, 0, 1)

        # define gradient function
        grad_fn_energy = jax.value_and_grad(
            jax.jit(self.loss_fn_energy), argnums=0, has_aux=True)
        
        # iterate through timesteps
        self.rng, rng_psi = jax.random.split(self.rng)

        @jax.jit
        def _through_time(inputs, t):
            state_energy, batch = inputs

            # compute gradient
            (loss_energy, (loss_psi, predicted)
            ), grad_energy = grad_fn_energy(state_energy.params,
                                            rng_psi, batch, t)

            # apply gradient to energy optimizer
            state_energy = state_energy.apply_gradients(grads=grad_energy)

            # if no teacher-forcing, replace next overvation with predicted
            batch = jax.lax.cond(
                self.config_settings['teacher_forcing'], lambda x: x,
                lambda x: x.at[t+1].set(predicted), batch)

            return ((state_energy, batch),
                    (loss_energy, loss_psi))

        # iterate through timesteps
        (state, _), (
            loss, _) = jax.lax.scan(
                _through_time, (state, batch),
                jnp.arange(batch.shape[0] - 1))

        loss = jnp.sum(loss)

        return loss, state
    
    def get_potential(self, state: train_state.TrainState) -> Callable[[jnp.ndarray], jnp.ndarray]:
        return lambda x: self.model_potential.apply(
            {'params': state.params}, x)

    def get_beta(self, _) -> float:
        return 0.
    
    def get_interaction(self, _) -> Callable[[jnp.ndarray], jnp.ndarray]:
        return lambda _: 0.
    
class JKOnetVanilla(JKOnet):
    """
    A variant of the JKOnet model without the use of ICNN (Input Convex Neural Networks).
    """
    def _prepare_otmap(self) -> MLP:
        return MLP(self.otmap_config['model']['layers'])

class JKOnetMongeGap(JKOnetVanilla):
    """
    A variant of the JKOnet model without the use of ICNN (Input Convex Neural Networks) but instead using the Monge gap regularization.
    """
    def _loss_fn_otmap(self,
        params_otmap: Dict[str, Any],
        params_energy: Dict[str, Any],
        data: jnp.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute the loss function for the optimal transport map (OT map) with Monge gap regularization.

        Parameters
        ----------
        params_otmap : Dict[str, Any]
            The parameters for the OT map model.
        params_energy : Dict[str, Any]
            The parameters for the energy model.
        data : jnp.ndarray
            The input data to be transported by the OT map.

        Returns
        -------
        Tuple[float, jnp.ndarray]
            A tuple containing:
            - float: The total loss, including potential energy, squared deviation,
              and Monge gap regularization.
            - jnp.ndarray: The predicted output from the OT map for the given data.
        """
        predicted = jax.vmap(lambda x: jax.grad(
            self.model_otmap.apply, argnums=1)(
                {'params': params_otmap}, x))(data)

        # jko objective
        loss_e = self.model_potential.apply(
            {'params': params_energy}, predicted)
        loss_p = jnp.mean(jnp.sum((predicted - data) ** 2, axis=1))
        loss = loss_e + 1 / (2 * self.tau) * loss_p

        # monge gap regularization
        loss += self.config_settings['monge_gap_reg'] * monge_gap(data, predicted)

        return loss, predicted


def get_optimize_psi_fn(
    loss_fn_psi: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    optimizer_psi: optax.GradientTransformation,
    n_iter: int = 100,
    min_iter: int = 50,
    max_iter: int = 200,
    inner_iter: int = 10,
    threshold: float = 1e-5,
    fploop: bool = False
) -> Callable:
    @jax.jit
    def step_fn_fpl(
        params_energy: jnp.ndarray,
        params_psi: jnp.ndarray,
        opt_state_psi: optax.OptState,
        data: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def cond_fn(
            iteration: int,
            constants: Tuple[jnp.ndarray, jnp.ndarray],
            state: Tuple[jnp.ndarray, optax.OptState, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            _, _ = constants
            _, _, _, _, grad = state

            norm = sum(jax.tree_util.tree_leaves(
                jax.tree_map(jnp.linalg.norm, grad)))
            norm /= count_parameters(grad)

            return jnp.logical_or(iteration == 0,
                                  jnp.logical_and(jnp.isfinite(norm),
                                                  norm > threshold))

        def body_fn(
            iteration: int,
            constants: Tuple[jnp.ndarray, jnp.ndarray],
            state: Tuple[jnp.ndarray, optax.OptState, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            compute_error: jnp.ndarray
        ) -> Tuple[jnp.ndarray, optax.OptState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            params_energy, data = constants
            params_psi, opt_state_psi, loss_psi, predicted, _ = state

            (loss_jko, predicted), grad_psi = jax.value_and_grad(
                loss_fn_psi, argnums=0, has_aux=True)(
                    params_psi, params_energy, data)

            # apply optimizer update
            updates, opt_state_psi = optimizer_psi.update(
                grad_psi, opt_state_psi)
            params_psi = optax.apply_updates(params_psi, updates)

            loss_psi = jax.ops.index_update(
                loss_psi, jax.ops.index[iteration // inner_iter], loss_jko)
            return params_psi, opt_state_psi, loss_psi, predicted, grad_psi

        # create empty vectors for losses and predictions
        loss_psi = jnp.full(
            (jnp.ceil(max_iter / inner_iter).astype(int)), 0., dtype=float)
        predicted = jnp.zeros_like(data, dtype=float)

        # define states and constants
        state = params_psi, opt_state_psi, loss_psi, predicted, params_psi
        constants = params_energy, data

        # iteratively _ psi
        params_psi, _, loss_psi, predicted, _ = fixpoint_iter(
            cond_fn, body_fn, min_iter, max_iter, inner_iter, constants, state)

        return params_psi, predicted, loss_psi

    @jax.jit
    def step_fn(
        params_energy: jnp.ndarray,
        params_psi: jnp.ndarray,
        opt_state_psi: optax.OptState,
        data: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # iteratively optimize psi
        def apply_psi_update(
            state_psi: Tuple[jnp.ndarray, optax.OptState],
            i: int
        ) -> Tuple[Tuple[jnp.ndarray, optax.OptState], Tuple[jnp.ndarray, jnp.ndarray]]:

            params_psi, opt_state_psi = state_psi

            # compute gradient of jko step
            (loss_psi, predicted), grad_psi = jax.value_and_grad(
                loss_fn_psi, argnums=0, has_aux=True)(
                    params_psi, params_energy, data)

            # apply optimizer update
            updates, opt_state_psi = optimizer_psi.update(
                grad_psi, opt_state_psi)
            params_psi = optax.apply_updates(params_psi, updates)

            return (params_psi, opt_state_psi), (loss_psi, predicted)

        (params_psi, _), (loss_psi, predicted) = jax.lax.scan(
            apply_psi_update, (params_psi, opt_state_psi), jnp.arange(n_iter))
        return params_psi, predicted[-1], loss_psi

    if fploop:
        return step_fn_fpl
    else:
        return step_fn