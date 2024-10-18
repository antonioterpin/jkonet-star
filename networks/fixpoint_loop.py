"""
This module provides a backprop-friendly fixed point loop implementation for JAX.

Authors: Jonathan Heek, Marco Cuturi, Charlotte Bunne
Source: https://github.com/bunnech/jkonet
"""

# imports
import functools
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any



@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def fixpoint_iter(
    cond_fn: Callable[[int, Any, Any], bool],
    body_fn: Callable[[int, Any, Any, bool], Any],
    min_iterations: int,
    max_iterations: int,
    inner_iterations: int,
    constants: Any,
    state: Any
) -> Any:
    """
    Implementation of a backprop-friendly fixed point loop.

    Parameters
    ----------
    cond_fn : Callable[[int, Any, Any], bool]
        A function that determines whether the loop should continue based on the
        current iteration, constants, and state.
    body_fn : Callable[[int, Any, Any, bool], Any]
        A function that defines the operations to perform in each iteration. It
        takes the current iteration number, constants, state, and a boolean indicating
        whether to compute an error.
    min_iterations : int
        Lower bound on the total number of fixed point iterations.
    max_iterations : int
        Upper bound on the total number of fixed point iterations.
    inner_iterations : int
        Default number of iterations in the inner loop.
    constants : Any
        Constant parameters passed to the body function during the loop.
    state : Any
        The initial state of the loop.

    Returns
    -------
    Any
        The final state after the loop terminates.
    """

    force_scan = (min_iterations == max_iterations)

    compute_error_flags = jnp.arange(inner_iterations) == inner_iterations - 1

    def max_cond_fn(iteration_state):
        iteration, state = iteration_state
        return jnp.logical_and(iteration < max_iterations,
                               jnp.logical_or(iteration < min_iterations,
                                              cond_fn(iteration, constants,
                                                      state)))

    def unrolled_body_fn(iteration_state):
        def one_iteration(iteration_state, compute_error):
            iteration, state = iteration_state
            state = body_fn(iteration, constants, state, compute_error)
            iteration += 1
            return (iteration, state), None
        iteration_state, _ = jax.lax.scan(one_iteration, iteration_state,
                                          compute_error_flags)
        return (iteration_state, None) if force_scan else iteration_state

    if force_scan:
        (_, state), _ = jax.lax.scan(
            lambda carry, x: unrolled_body_fn(carry),
            (0, state), None,
            length=max_iterations // inner_iterations)
    else:
        _, state = jax.lax.while_loop(
            max_cond_fn, unrolled_body_fn, (0, state))
    return state


def fixpoint_iter_fwd(
    cond_fn: Callable[[int, Any, Any], bool],
    body_fn: Callable[[int, Any, Any, bool], Any],
    min_iterations: int,
    max_iterations: int,
    inner_iterations: int,
    constants: Any,
    state: Any
) -> Tuple[Any, Tuple[Any, int, Any]]:
    """
    Forward pass for the fixed-point iteration loop, storing intermediate states.

    Parameters
    ----------
    cond_fn : Callable[[int, Any, Any], bool]
        A function that determines whether the loop should continue based on the
        current iteration, constants, and state.
    body_fn : Callable[[int, Any, Any, bool], Any]
        A function that defines the operations to perform in each iteration. It
        takes the current iteration number, constants, state, and a boolean indicating
        whether to compute an error.
    min_iterations : int
        Lower bound on the total number of fixed point iterations.
    max_iterations : int
        Upper bound on the total number of fixed point iterations.
    inner_iterations : int
        Default number of iterations in the inner loop.
    constants : Any
        Constant parameters passed to the body function during the loop.
    state : Any
        The initial state of the loop.

    Returns
    -------
    Tuple[Any, Tuple[Any, int, Any]]
        The final state after the loop terminates, and a tuple containing the
        constants, the final iteration number, and the recorded intermediate states.
    """

    force_scan = (min_iterations == max_iterations)

    compute_error_flags = jnp.arange(inner_iterations) == inner_iterations - 1

    states = jax.tree_util.tree_map(lambda x: jnp.zeros(
        (max_iterations // inner_iterations + 1,) + x.shape,
        dtype=x.dtype), state)

    def max_cond_fn(iteration_states_state):
        iteration, _, state = iteration_states_state
        return jnp.logical_and(iteration < max_iterations,
                               jnp.logical_or(iteration < min_iterations,
                                              cond_fn(iteration, constants,
                                                      state)))

    def unrolled_body_fn(iteration_states_state):
        iteration, states, state = iteration_states_state
        states = jax.tree_util.tree_multimap(
            lambda states, state: jax.lax.dynamic_update_index_in_dim(
                states, state, iteration // inner_iterations, 0),
            states, state)

        def one_iteration(iteration_state, compute_error):
            iteration, state = iteration_state
            state = body_fn(iteration, constants, state, compute_error)
            iteration += 1
            return (iteration, state), None

        iteration_state, _ = jax.lax.scan(one_iteration, (iteration, state),
                                          compute_error_flags)
        iteration, state = iteration_state
        out = (iteration, states, state)
        return (out, None) if force_scan else out

    if force_scan:
        (iteration, states, state), _ = jax.lax.scan(
            lambda carry, x: unrolled_body_fn(carry),
            (0, states, state), None,
            length=max_iterations // inner_iterations)
    else:
        iteration, states, state = jax.lax.while_loop(
            max_cond_fn, unrolled_body_fn, (0, states, state))

    return state, (constants, iteration, states)


def fixpoint_iter_bwd(
    cond_fn: Callable[[int, Any, Any], bool],
    body_fn: Callable[[int, Any, Any, bool], Any],
    min_iterations: int,
    max_iterations: int,
    inner_iterations: int,
    res: Tuple[Any, int, Any],
    g: Any
) -> Tuple[Any, Any]:
    """
    Backward pass for the fixed-point iteration loop.

    Parameters
    ----------
    cond_fn : Callable[[int, Any, Any], bool]
        A function that was used in the forward pass to determine whether the loop should continue.
    body_fn : Callable[[int, Any, Any, bool], Any]
        A function that defines the operations performed in each iteration during the forward pass.
    min_iterations : int
        The minimum number of iterations that was performed in the forward pass.
    max_iterations : int
        The maximum number of iterations that was performed in the forward pass.
    inner_iterations : int
        The number of iterations in each inner loop during the forward pass.
    res : Tuple[Any, int, Any]
        A tuple containing the constants, final iteration count, and recorded intermediate states from the forward pass.
    g : Any
        The gradient with respect to the final state.

    Returns
    -------
    Tuple[Any, Any]
        A tuple containing the gradients with respect to the constants and the initial state.
    """

    del cond_fn

    force_scan = (min_iterations == max_iterations)
    constants, iteration, states = res

    g_constants = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=x.dtype)
                               if isinstance(x, jnp.ndarray) else 0, constants)

    def bwd_cond_fn(iteration_g_gconst):
        iteration, _, _ = iteration_g_gconst
        return iteration >= 0

    def unrolled_body_fn_no_errors(iteration, constants, state):
        compute_error_flags = jnp.zeros((inner_iterations,), dtype=bool)

        def one_iteration(iteration_state, compute_error):
            iteration, state = iteration_state
            state = body_fn(iteration, constants, state, compute_error)
            iteration += 1
            return (iteration, state), None

        iteration_state, _ = jax.lax.scan(one_iteration, (iteration, state),
                                          compute_error_flags)
        _, state = iteration_state
        return state

    def unrolled_body_fn(iteration_g_gconst):
        iteration, g, g_constants = iteration_g_gconst
        state = jax.tree_util.tree_map(lambda x: x[iteration // inner_iterations],
                             states)
        _, pullback = jax.vjp(unrolled_body_fn_no_errors, iteration, constants,
                              state)
        _, gi_constants, g_state = pullback(g)
        g_constants = jax.tree_util.tree_multimap(
            lambda x, y: x + jax.lax.convert_element_type(
                y, jnp.array(x).dtype), g_constants, gi_constants)
        g_state = jax.tree_util.tree_multimap(
            lambda g1, g2: jax.lax.convert_element_type(g1, g2.dtype),
            g_state, g)
        out = (iteration - inner_iterations, g_state, g_constants)
        return (out, None) if force_scan else out

    if force_scan:
        (_, g_state, g_constants), _ = jax.lax.scan(
            lambda carry, x: unrolled_body_fn(carry),
            (0, g, g_constants), None,
            length=max_iterations // inner_iterations)
    else:
        # BUG: ValueError: setting an array element with a sequence.
        _, g_state, g_constants = jax.lax.while_loop(
            bwd_cond_fn, unrolled_body_fn,
            (iteration - inner_iterations, g, g_constants))

    return g_constants, g_state


fixpoint_iter = jax.custom_vjp(fixpoint_iter,
                               nondiff_argnums=(0, 1, 2, 3, 4))

fixpoint_iter.defvjp(fixpoint_iter_fwd, fixpoint_iter_bwd)
