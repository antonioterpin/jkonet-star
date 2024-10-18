"""
Module for gradient computation and parameter analysis in Flax neural networks using JAX.

Functions
---------

- ``network_grad``
    Computes the gradient of the network's output with respect to its input. The gradient is evaluated for each sample using vectorized mapping (vmap).
    
- ``network_grad_time``
    Computes the gradient of the network's output with respect to its input, excluding the time component.
    
- ``count_parameters``
    Returns the total number of parameters in the given Flax neural network model.
"""


import jax
from typing import Callable, Dict
import jax.numpy as jnp
import flax.linen as nn

def network_grad(network: nn.Module, params: Dict[str, jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Computes the gradient of the network's output with respect to its input for each sample.

    Parameters
    ----------
    network : nn.Module
        The Flax neural network module.
    params : Dict[str, jnp.ndarray]
        Dictionary containing model parameters.

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        A function that computes gradients with respect to the network's input.
    """
    return jax.vmap(lambda v: jax.grad(network.apply, argnums=1)({'params': params}, v))

def network_grad_time(network: nn.Module, params: Dict[str, jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Computes the gradient of the network's output with respect to the input, excluding the time component.

    In the time-varying JKOnet* model, the gradient in the loss is computed with respect to the input, excluding the time component.

    Parameters
    ----------
    network : nn.Module
        The Flax neural network module.
    params : Dict[str, jnp.ndarray]
        Dictionary containing model parameters.

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        A function that computes gradients with respect to the input, excluding the time component.
    """
    def grad_fn(v):
        partial_v = v[:-1]
        def loss_fn(partial_input):
            full_input = jax.numpy.concatenate([partial_input, v[-1:]], axis=-1)
            return network.apply({'params': params}, full_input)
        return jax.grad(loss_fn)(partial_v)
    return jax.vmap(grad_fn, in_axes=0)

def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of parameters in the model.

    Parameters
    ----------
    model : nn.Module
        The Flax neural network module.

    Returns
    -------
    int
        The total number of parameters in the model.
    """
    return sum(map(lambda x: x.size, jax.tree_flatten(model)[0]))