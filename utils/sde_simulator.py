"""
This module provides tools to simulate Stochastic Differential Equations (SDEs) using explicit and implicit schemes. 
It includes functionality to generate trajectories based on specified models, potentials, internal energies, and interactions. 
The simulations are performed using ``JAX``, which enables efficient computations and automatic differentiation.

Classes
-----------
- ``SDESimulator``
    Simulates SDEs using an explicit scheme, allowing for the application of potential, internal energy, and interaction 
    components. The class supports forward sampling of trajectories based on the initial condition and a JAX random key.

- ``SDESimulator_implicit_time``
    Simulates SDEs using implicit time-stepping methods. This class is designed to handle time-varying potentials 
    and performs fixed-point iterations to account for implicit dynamics. It supports forward sampling similar to ``SDESimulator``.

Functions
-------------
- ``get_SDE_predictions``
    A helper function to choose between explicit and implicit SDE simulators based on the model type and return 
    the simulated trajectories.

Usage example
-------------
To simulate trajectories using an explicit SDE simulator:

    >>> import jax.numpy as jnp
    >>> import jax.random as jrandom
    >>> key = jrandom.PRNGKey(0)
    >>> init_pp = jnp.array([[0.0, 0.0]])
    >>> sde_simulator = SDESimulator(dt=0.01, n_timesteps=100, start_timestep=0, potential=False, internal=0.1, interaction=False)
    >>> trajectories = sde_simulator.forward_sampling(key, init_pp)
    >>> print(trajectories.shape)  # Output: (101, 1, 2)

For implicit time-stepping with a time-dependent potential:

    >>> potential_func = lambda x: 0.5 * jnp.sum(jnp.square(x))
    >>> implicit_simulator = SDESimulator_implicit_time(dt=0.01, n_timesteps=100, start_timestep=0, potential=potential_func, internal=False, interaction=False)
    >>> trajectories = implicit_simulator.forward_sampling(key, init_pp)
    >>> print(trajectories.shape)  # Output: (101, 1, 2)

References
----------
- `jax`: https://github.com/google/jax
- Stochastic Differential Equations (SDE): https://en.wikipedia.org/wiki/Stochastic_differential_equation
"""


import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Callable, Union

def get_SDE_predictions(
    model: str,
    dt: float,
    n_timesteps: int,
    start_timestep: int,
    potential: Union[bool, Callable[[jnp.ndarray], jnp.ndarray]],
    internal: Union[bool, Callable[[jnp.ndarray], jnp.ndarray], float],
    interaction: Union[bool, Callable[[jnp.ndarray], jnp.ndarray]],
    key: jax.random.PRNGKey,
    init_pp: jnp.ndarray
) -> jnp.ndarray:
    """
    Get predictions from a Stochastic Differential Equation (SDE) simulator based on the specified model type.

    Depending on the model type, it selects the appropriate SDE simulator (`SDESimulator` or `SDESimulator_implicit_time`)
    and performs forward sampling to generate predictions.

    Parameters
    ----------
    model : str
        The name of the model to use for simulation. If 'jkonet-star-time-potential' is specified,
        `SDESimulator_implicit_time` is used; otherwise, `SDESimulator` is used.
    dt : float
        The timestep size for the SDE simulation.
    n_timesteps : int
        The total number of timesteps to simulate.
    start_timestep : int
        The initial timestep index for the simulation.
    potential : Union[bool, Callable[[jnp.ndarray], jnp.ndarray]]
        If `True`, a potential function is used in the simulation. If a callable is provided, it should accept
        a JAX array as input and return the potential. If `False`, no potential is applied.
    internal : Union[bool, Callable[[jnp.ndarray], jnp.ndarray], float]
        If a float, represents the internal energy scale used in the simulation. If a callable is provided, it should
        accept a JAX array, returning the internal component. If `False`, no internal component is used.
    interaction : Union[bool, Callable[[jnp.ndarray], jnp.ndarray]]
        If `True`, an interaction function is used in the simulation. If a callable is provided, it should accept
        a JAX array as input and return the interaction component. If `False`, no interaction is applied.
    key : jax.random.PRNGKey
        A JAX random key used for stochastic processes in the SDE simulation.
    init_pp : jnp.ndarray
        The initial state for the simulation, typically a JAX array representing the starting point of the system.

    Returns
    -------
    jnp.ndarray
        An array representing the simulated trajectories of the system, with shape (n_timesteps + 1, ...),
        where the first dimension corresponds to the timesteps and the remaining dimensions correspond
        to the state variables of the system.
    """
    if model == 'jkonet-star-time-potential':
        sde = SDESimulator_implicit_time
    else:
        sde = SDESimulator
    return sde(dt, n_timesteps, start_timestep, potential, internal, interaction).forward_sampling(key, init_pp)

class SDESimulator:
    """
    Simulator for Stochastic Differential Equations (SDEs) with an explicit scheme.

    Parameters
    ----------
    dt : float
        The timestep size for the simulation.

    n_timesteps : int
        The number of timesteps to simulate.

    start_timestep : int
        The initial timestep index for the simulation.

    potential : Optional[Callable[[jnp.ndarray, jnp.ndarray]
        If a callable, it should take a JAX array as input and return the potential. If `False`,
        no potential is applied.

    internal : Optional[Callable[[jnp.ndarray, jnp.ndarray]
        If a callable, it should take a JAX array as input and return the internal component. If `False`,
        no internal component is used.

    interaction : Optional[Callable[[jnp.ndarray, jnp.ndarray]
        If a callable, it should take a JAX array as input and return the interaction component. If `False`,
        no interaction is applied.
    Methods
    -------
    forward_sampling(key: jax.random.PRNGKey, init: jnp.ndarray) -> jnp.ndarray
        Performs forward sampling of the SDE from the initial condition `init` using the provided random key.

    """
    def __init__(
            self, 
            dt: float,
            n_timesteps: int,
            start_timestep: int,
            potential: Union[bool, Callable[[jnp.ndarray], jnp.ndarray]],
            internal: Union[bool, Callable[[jnp.ndarray], jnp.ndarray], float],
            interaction: Union[bool, Callable[[jnp.ndarray], jnp.ndarray]]):

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
            
        
        def forward_sampling(key: jax.random.PRNGKey, init: jnp.ndarray) -> jnp.ndarray:
            """
            Performs forward sampling of the SDE from the initial condition.

            Parameters
            ----------
            key : jax.random.PRNGKey
                Random key used for sampling.

            init : jnp.ndarray
                Initial condition for the simulation.

            Returns
            -------
            jnp.ndarray
                The array of simulated trajectories with shape (n_timesteps + 1, ...) where
                the first dimension represents the timestep and the remaining dimensions
                represent the state variables.
            """
            pp = jnp.copy(init)
            trajectories = [pp]
            for i in range(1, n_timesteps + 1):
                key, subkey = jrandom.split(key, 2)
                pp = pp + potential_component(pp, subkey) + internal_component(pp, subkey) + interaction_component(pp, subkey)
                trajectories.append(pp)
            return jnp.asarray(trajectories)

        self.forward_sampling = jax.jit(forward_sampling)


class SDESimulator_implicit_time:
    """
    Simulator for Stochastic Differential Equations (SDEs) using implicit methods.

    Parameters
    ----------
    dt : float
        The timestep size for the simulation.

    n_timesteps : int
        The number of timesteps to simulate.

    start_timestep : int
        The initial timestep index for the simulation. In the case that we are working with time-varying potentials
        the start time of the simulation is necessary.

    potential : Optional[Callable[[jnp.ndarray, jnp.ndarray]
        If a callable, it should take a JAX array and a time array
        as input and return the potential. If `False`, no potential is applied.

    internal : Optional[Callable[[jnp.ndarray, jnp.ndarray]
        If a callable, it should take a JAX array as input and return the internal component. If `False`,
        no internal component is used.

    interaction : Optional[Callable[[jnp.ndarray, jnp.ndarray]
        If a callable, it should take a JAX array as input and return the interaction component. If `False`,
        no interaction is applied.

    Methods
    -------
    forward_sampling(key: jax.random.PRNGKey, init: jnp.ndarray) -> jnp.ndarray
        Performs forward sampling of the SDE from the initial condition `init` using the provided random key.
    """

    def __init__(
            self,
            dt: float,
            n_timesteps: int,
            start_timestep: int,
            potential: Union[bool, Callable[[jnp.ndarray], jnp.ndarray]],
            internal: Union[bool, Callable[[jnp.ndarray], jnp.ndarray], float],
            interaction: Union[bool, Callable[[jnp.ndarray], jnp.ndarray]]):
        self.dt = dt
        self.n_timesteps = n_timesteps
        self.potential = potential
        self.sqrtdt = jnp.sqrt(2 * dt)

        def potential_component_implicit(pp: jnp.ndarray, t_array: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
            """
            Computes the implicit potential component using fixed-point iterations.

            Parameters
            ----------
            pp : jnp.ndarray
                The current state of the simulation.

            t_array : jnp.ndarray
                The time array for the current step.

            key : jax.random.PRNGKey
                Random key used for sampling.

            Returns
            -------
            jnp.ndarray
                The implicit potential component to be added to the state.
            """
            if self.potential:
                def fixed_point_iteration(x, pp, t_array):
                    concat_pos_time = jnp.concatenate([x, t_array], axis=-1)
                    gradient = jax.grad(potential)(concat_pos_time)
                    return pp - gradient[..., :-1] * dt

                # Initial guess for implicit method
                x = pp
                for _ in range(50):  # Perform fixed-point iterations
                    x = fixed_point_iteration(x, pp, t_array)

                return x - pp
            else:
                return jnp.zeros(pp.shape)

        def forward_sampling(key, init):
            """
            Performs forward sampling of the SDE from the initial condition.

            Parameters
            ----------
            key : jax.random.PRNGKey
                Random key used for sampling.

            init : jnp.ndarray
                Initial condition for the simulation.

            Returns
            -------
            jnp.ndarray
                The array of simulated trajectories with shape (n_timesteps + 1, ...) where
                the first dimension represents the timestep and the remaining dimensions
                represent the state variables.
            """
            pp = jnp.copy(init)
            trajectories = [pp]
            for i in range(start_timestep, start_timestep + n_timesteps):
                # for i in range(start_timestep, start_timestep + n_timesteps * timestep, timestep):
                key, subkey = jrandom.split(key, 2)
                t_array = (i) * jnp.ones((pp.shape[0], 1))  # Create time array for current step
                pp = pp + potential_component_implicit(pp, t_array, subkey)
                trajectories.append(pp)
            return jnp.asarray(trajectories)

        self.forward_sampling = jax.jit(forward_sampling)
