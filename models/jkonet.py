# Implementation of JKOnet see https://arxiv.org/abs/2106.06345
# Monge gap regularizer see https://arxiv.org/abs/2302.04953
import jax
import jax.numpy as jnp
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

class JKOnet(LearningDiffusionModel):
    def load_dataset(self, dataset_name: str):
        return PopulationDataset(dataset_name)
    
    def __init__(self, config, data_dim, tau) -> None:
        super().__init__()
        self.tau = tau
        self.data_dim = data_dim
        self.potential_optimizer = config['energy']['optim']
        self.model_potential = MLP(config['energy']['model']['layers'])

        # otmap
        self.config_settings = config['settings']
        self.otmap_config = config['otmap']
        self.otmap_optimizer = get_optimizer(config['otmap']['optim'])

    def _loss_fn_otmap(self, params_otmap, params_energy, data):
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

    def _prepare_otmap(self):
        return ICNN(dim_hidden=self.otmap_config['model']['layers'],
                    init_fn=self.otmap_config['model']['init_fn'],
                    pos_weights=self.otmap_config['model']['pos_weights'])

    def create_state(self, rng):
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
    
    def create_state_from_params(self, params):
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
    def loss_fn_energy(self, params_energy, rng_psi, batch, t):
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
    
    def train_step(self, state, sample):
        batch = jnp.stack(sample, axis=2).transpose(2, 0, 1)

        # define gradient function
        grad_fn_energy = jax.value_and_grad(
            jax.jit(self.loss_fn_energy), argnums=0, has_aux=True)
        
        # iterate through time steps
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

        # iterate through time steps
        (state, _), (
            loss, _) = jax.lax.scan(
                _through_time, (state, batch),
                jnp.arange(batch.shape[0] - 1))

        loss = jnp.sum(loss)

        return loss, state
    
    def get_potential(self, state):
        return lambda x: self.model_potential.apply(
            {'params': state.params}, x)

    def get_beta(self, _):
        return 0.
    
    def get_interaction(self, _):
        return lambda _: 0.
    
class JKOnetVanilla(JKOnet):
    def _prepare_otmap(self):
        return MLP(self.otmap_config['model']['layers'])

class JKOnetMongeGap(JKOnetVanilla):
    def _loss_fn_otmap(self, params_otmap, params_energy, data):
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


# Source: https://github.com/bunnech/jkonet
def get_optimize_psi_fn(loss_fn_psi, 
                        optimizer_psi, n_iter=100,
                        min_iter=50, max_iter=200, inner_iter=10,
                        threshold=1e-5,
                        fploop=False):
    """Create a training function of Psi."""

    @jax.jit
    def step_fn_fpl(params_energy, params_psi, opt_state_psi, data):
        def cond_fn(iteration, constants, state):
            """Condition function for optimization of convex potential Psi.
            """
            _, _ = constants
            _, _, _, _, grad = state

            norm = sum(jax.tree_util.tree_leaves(
                jax.tree_map(jnp.linalg.norm, grad)))
            norm /= count_parameters(grad)

            return jnp.logical_or(iteration == 0,
                                  jnp.logical_and(jnp.isfinite(norm),
                                                  norm > threshold))

        def body_fn(iteration, constants, state, compute_error):
            """Body loop for gradient update of convex potential Psi.
            """
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
    def step_fn(params_energy, params_psi, opt_state_psi, data):
        # iteratively optimize psi
        def apply_psi_update(state_psi, i):
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