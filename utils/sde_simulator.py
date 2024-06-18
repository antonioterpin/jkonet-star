import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Callable, Union

class SDESimulator:
    """
    Simulator for SDEs.

    Usage:
    >>> simulator = SDESimulator(dt, n_timesteps, potential, internal, interaction)
    >>> simulator.forward_sampling(key, init)
    """
    def __init__(
            self, 
            dt: float,
            n_timesteps: int,
            potential: Union[bool, Callable], 
            internal: Union[bool, Callable, float],
            interaction: Union[bool, Callable]):

        sqrtdt = jnp.sqrt(2 * dt)
        potential_component = lambda pp, key: jnp.zeros(pp.shape)
        internal_component = lambda pp, key: jnp.zeros(pp.shape)
        interaction_component = lambda pp, key: jnp.zeros(pp.shape)

        if potential:
            potential_grad = jax.grad(potential)
            flow = jax.vmap(lambda v: -potential_grad(v))
            potential_component = lambda pp, key: flow(pp) * dt

        if internal:
            # At the moment we use wiener process
            if not isinstance(internal, float):
                raise NotImplementedError(
                    'Generic internal energies not implemented yet.')
            
            internal_component = lambda pp, key: -jnp.sqrt(jnp.abs(internal)) * jrandom.normal(key, shape=pp.shape) * sqrtdt

        if isinstance(interaction, Callable):
            interaction_grad = jax.vmap(lambda v: jax.grad(interaction)(v))
            def get_interaction_component(pp):
                return lambda p: jnp.mean(-interaction_grad(p -  pp), axis=0)
            interaction_component = lambda pp, _: jax.vmap(get_interaction_component(pp))(pp) * dt
            
        
        def forward_sampling(key, init):
            pp = jnp.copy(init)
            trajectories = [pp]
            for i in range(1, n_timesteps + 1):
                key, subkey = jrandom.split(key, 2)
                pp = pp + potential_component(pp, subkey) + internal_component(pp, subkey) + interaction_component(pp, subkey)
                trajectories.append(pp)
            return jnp.asarray(trajectories)

        self.forward_sampling = jax.jit(forward_sampling)