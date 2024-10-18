"""
This module contains optimization utils used to train the models.

Source: https://github.com/bunnech/jkonet

Functions
---------
- ``get_optimizer``
    Returns an Optax optimizer object based on the provided configuration.

- ``create_train_state``
    Creates an initial `TrainState` for the given model and optimizer.

- ``create_train_state_from_params``
    Creates a `TrainState` from existing model parameters.

- ``global_norm``
    Computes the global norm of gradients across a nested structure of tensors.

- ``clip_weights_icnn``
    Clip the weights of an Input Convex Neural Network (ICNN).
"""
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
from flax.core import freeze, FrozenDict
import optax
from typing import Dict, Any
import chex


def get_optimizer(config: Dict[str, Any]) -> optax.GradientTransformation:
    """
    Returns an Optax optimizer object based on the provided configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary containing optimizer configuration. Expected keys are:

        - 'optimizer': The name of the optimizer ('Adam' or 'SGD').
        - 'lr': Learning rate for the optimizer.
        - 'beta1': Beta1 parameter for the Adam optimizer.
        - 'beta2': Beta2 parameter for the Adam optimizer.
        - 'eps': Epsilon parameter for the Adam optimizer.
        - 'grad_clip': Optional maximum global norm for gradient clipping.

    Returns
    -------
    optax.GradientTransformation
        The configured Optax optimizer object.

    Raises
    ------
    NotImplementedError
        If the optimizer name is not supported.
    """

    chex.assert_type([config['lr'], config['beta1'], config['beta2'], config['eps']], [float] * 4)
    chex.assert_scalar_positive(config['lr'])
    chex.assert_scalar_positive(config['beta1'])
    chex.assert_scalar_positive(config['beta2'])
    chex.assert_scalar_positive(config['eps'])

    if 'grad_clip' in config and config['grad_clip'] is not None:
        chex.assert_type(config['grad_clip'], float)
        chex.assert_scalar_positive(config['grad_clip'])

    optimizer_name = config['optimizer']
    if optimizer_name == 'Adam':
        optimizer = optax.adam(learning_rate=config['lr'],
                               b1=config['beta1'], b2=config['beta2'],
                               eps=config['eps'])
    elif optimizer_name == 'SGD':
        optimizer = optax.sgd(learning_rate=config['lr'],
                              momentum=None, nesterov=False)
    else:
        raise NotImplementedError(
            f'Optimizer {optimizer_name} not supported yet!')

    if config['grad_clip']:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config['grad_clip']),
            optimizer)
    return optimizer


def create_train_state(
        rng: jax.random.PRNGKey,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        input_shape: int
) -> train_state.TrainState:
    """
    Creates an initial `TrainState` for the given model and optimizer.

    Parameters
    ----------
    rng : jax.random.PRNGKey
        Random key used for initializing the model parameters.
    model : nn.Module
        Flax model used for creating the initial state.
    optimizer : optax.GradientTransformation
        Optimizer object used for updating the model parameters.
    input_shape : int
        Shape of the input data used to initialize the model.

    Returns
    -------
    train_state.TrainState
        The initialized train state containing model parameters and optimizer.
    """

    params = model.init(rng, jnp.ones(input_shape))['params']
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

def create_train_state_from_params(
        model: nn.Module,
        params: Dict[str, Any],
        optimizer: optax.GradientTransformation
) -> train_state.TrainState:
    """
    Creates a `TrainState` from existing model parameters.

    Parameters
    ----------
    model : nn.Module
        Flax model used for creating the initial state.
    params : Dict[str, Any]
        Dictionary of model parameters.
    optimizer : optax.GradientTransformation
        Optimizer object used for updating the model parameters.

    Returns
    -------
    train_state.TrainState
        The train state containing the provided model parameters and optimizer.
    """
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


def global_norm(updates: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the global norm of gradients across a nested structure of tensors.

    Parameters
    ----------
    updates : Dict[str, jnp.ndarray]
        Dictionary where values are tensors (e.g., gradients).

    Returns
    -------
    jnp.ndarray
        The global norm of the gradients.
    """
    return jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(updates)]))


def clip_weights_icnn(params: FrozenDict) -> FrozenDict:
    """
    Clip the weights of an Input Convex Neural Network (ICNN).

    This function modifies the weights of the ICNN by clipping the values in kernels that start with 'Wz'
    to ensure they are non-negative. This is necessary to maintain the convexity property of the ICNN.

    Parameters
    ----------
    params : FrozenDict
        A frozen dictionary containing the parameters of the ICNN.

    Returns
    -------
    Any
        A frozen dictionary with the same structure as `params`, but with the relevant weights clipped to be non-negative.
    """
    params = params.unfreeze()
    for k in params.keys():
        if (k.startswith('Wz')):
            params[k]['kernel'] = jnp.clip(params[k]['kernel'], a_min=0)

    return freeze(params)


def penalize_weights_icnn(params: FrozenDict) -> jnp.ndarray:
    """
    Compute a penalty for negative weights in an ICNN.

    This function calculates a penalty term based on the L2 norm of any negative values in the weights
    that start with 'Wz'. This penalty can be added to the loss function during training to encourage
    the network to maintain non-negative weights in those layers, which is important for the ICNN's convexity.

    Parameters
    ----------
    params : FrozenDict
        A frozen dictionary containing the parameters of the ICNN.

    Returns
    -------
    jnp.ndarray
        A scalar penalty value representing the sum of the L2 norms of the negative weights.
    """
    penalty = 0
    for k in params.keys():
        if (k.startswith('Wz')):
            penalty += jnp.linalg.norm(jax.nn.relu(-params[k]['kernel']))
    return penalty
