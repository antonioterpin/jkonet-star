"""
This module provides functions for computing Wasserstein distances, Sinkhorn divergences, and optimal transport
couplings between two sets of points using the POT (Python Optimal Transport) and JAX-OTT libraries.

Functions
---------
- ``wasserstein_couplings``
    Computes the optimal transport plan (couplings) between two sets of points `xs` and `ys` using the POT library.

- ``wasserstein_loss``
    Computes the Wasserstein loss between two sets of points `xs` and `ys`, which quantifies the cost of transporting
    one distribution to match the other.

- ``sinkhorn_loss``
    Computes the Sinkhorn divergence, a regularized Wasserstein distance, between two sets of points `xs` and `ys`
    using the JAX-OTT library.

- ``compute_couplings``
    Computes the couplings between particles in two consecutive batches using the Wasserstein transport plan.

- ``compute_couplings_sinkhorn``
    Computes the Sinkhorn couplings between particles in two consecutive batches using the JAX-OTT library.

- ``compute_relevant_couplings``
    Computes the relevant couplings between particles in two consecutive batches by filtering out the couplings with low weights.

Libraries used
--------------
- ``JAX``: A framework for high-performance machine learning and numerical computations with automatic differentiation.
- ``POT``: A Python library for optimal transport calculations.
- ``JAX-OTT``: A library for computing optimal transport using JAX.
- ``Chex``: A library providing assertions and utilities for JAX-related computations.

References
----------
- ``POT`` library documentation: https://pythonot.github.io/
- ``JAX-OTT`` library documentation: https://ott-jax.readthedocs.io/
"""

import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
import ot
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import chex

def wasserstein_couplings(xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    """
    This function uses the POT (Python Optimal Transport) to compute the optimal transport plan (couplings)
    between two sets of points xs and ys.

    Parameters
    ----------
    xs : jnp.ndarray
        An array of shape (n_samples_x, n_features) representing the first set of points.
    ys : jnp.ndarray
        An array of shape (n_samples_y, n_features) representing the second set of points.

    Returns
    -------
    jnp.ndarray
        The optimal transport plan matrix of shape (n_samples_x, n_samples_y).

    References
    ----------
    - POT library documentation: https://pythonot.github.io/

    Example
    -------
    >>> import jax.numpy as jnp
    >>> import ot
    >>> xs = jnp.array([[0., 0.], [1., 0.]])
    >>> ys = jnp.array([[0., 1.], [2., 2.]])
    >>> wasserstein_couplings(xs, ys)
    DeviceArray([[0.5, 0. ],
                 [0. , 0.5]], dtype=float32)
    """

    chex.assert_rank(xs, 2)
    chex.assert_rank(ys, 2)
    chex.assert_axis_dimension(xs, axis=1, expected=ys.shape[1])
    chex.assert_type([xs, ys], float)

    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    M = ot.dist(xs, ys)

    return ot.emd(a, b, M, numItermax=1000000)

def wasserstein_loss(xs: jnp.ndarray, ys: jnp.ndarray, wasserstein_metric: int) -> jnp.ndarray:
    """
    Computes the Wasserstein loss between two sets of points `xs` and `ys`.

    The Wasserstein loss quantifies the cost of transporting the distribution of points in `xs`
    to match the distribution of points in `ys`. Since the distance is calculated using the 'sqeuclidean',
    it computes the W2 error.

    This function uses the POT (Python Optimal Transport) library.

    Parameters
    ----------
    xs : jnp.ndarray
        An array of shape (n_samples_x, n_features) representing the first set of points.
    ys : jnp.ndarray
        An array of shape (n_samples_y, n_features) representing the second set of points.

    Returns
    -------
    jnp.ndarray
        A scalar representing the Wasserstein loss between the two distributions.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> import ot
    >>> xs = jnp.array([[0., 0.], [1., 0.]])
    >>> ys = jnp.array([[0., 1.], [2., 2.]])
    >>> wasserstein_loss(xs, ys)
    DeviceArray(3.0, dtype=float32)

    References
    ----------
    - POT library documentation: https://pythonot.github.io/
    """
    chex.assert_rank(xs, 2)
    chex.assert_rank(ys, 2)
    chex.assert_axis_dimension(xs, axis=1, expected=ys.shape[1])
    chex.assert_type([xs, ys], float)
    chex.assert_scalar_in(wasserstein_metric, 1, 2)

    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    if wasserstein_metric == 1:
        M = ot.dist(xs, ys, metric='euclidean')
    else:
        M = ot.dist(xs, ys)

    return ot.emd2(a, b, M, numItermax=1000000)

@jax.jit
def sinkhorn_loss(xs: jnp.ndarray, ys: jnp.ndarray, epsilon: float = 1.0) -> float:
    """
    Computes the Sinkhorn divergence (a regularized Wasserstein distance) between two sets of points `xs` and `ys`.

    This function uses the JAX-OTT (Optimal Transport Tools) library to compute the Sinkhorn divergence,
    which is a regularized version of the Wasserstein distance.

    Parameters
    ----------
    xs : jnp.ndarray
        An array of shape (n_samples_x, n_features) representing the first set of points.
    ys : jnp.ndarray
        An array of shape (n_samples_y, n_features) representing the second set of points.
    epsilon : float, optional
        Regularization parameter for the Sinkhorn algorithm, by default 1.

    Returns
    -------
    float
        The Sinkhorn divergence between the two sets of points.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> from ott.geometry import pointcloud
    >>> from ott.problems.linear import linear_problem
    >>> from ott.solvers.linear import sinkhorn
    >>> xs = jnp.array([[0., 0.], [1., 0.]])
    >>> ys = jnp.array([[0., 1.], [2., 2.]])
    >>> sinkhorn_loss(xs, ys, epsilon=0.1)
    DeviceArray(3.0693126, dtype=float32)

    References
    ----------
    - JAX-OTT library documentation: https://ott-jax.readthedocs.io/

    """
    chex.assert_rank(xs, 2)
    chex.assert_rank(ys, 2)
    chex.assert_axis_dimension(xs, axis=1, expected=ys.shape[1])
    chex.assert_type([xs, ys], float)

    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    geom = pointcloud.PointCloud(xs, ys, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a, b)

    solver = sinkhorn.Sinkhorn()
    out = solver(prob)

    return out.reg_ot_cost

def compute_couplings_sinkhorn(batch: jnp.ndarray, batch_next: jnp.ndarray, time: int, epsilon: float = 1.0) -> jnp.ndarray:
    """
    Computes the Sinkhorn couplings (a regularized Wasserstein distance) between two sets of points `batch` and `batch_next`.

    This function uses the JAX-OTT (Optimal Transport Tools) library to compute the Sinkhorn divergence, which is a regularized version of the Wasserstein distance.

    Parameters
    ----------
    batch : jnp.ndarray
        The array of particles at the current timestep with shape (n_particles, n_features).

    batch_next : jnp.ndarray
        The array of particles at the next timestep with shape (n_particles, n_features).

    time : int
        The timestep of batch_next.

    epsilon : float, optional
        Regularization parameter for the Sinkhorn algorithm, by default 1.

    Returns
    -------
    jnp.ndarray
        An array of shape (n_relevant_couplings, 2 * n_features + 2) where each row contains:

        - Particle from `batch` (shape: (n_features,))
        - Particle from `batch_next` (shape: (n_features,))
        - Time (float)
        - Coupling weight (float)

        Only the relevant couplings, where the weight is greater than a threshold, are included.
    """
    chex.assert_rank(batch, 2)
    chex.assert_rank(batch_next, 2)
    chex.assert_axis_dimension(batch, axis=1, expected=batch_next.shape[1])
    chex.assert_type([batch, batch_next], float)
    chex.assert_type(time, int)

    a = jnp.ones(batch.shape[0]) / batch.shape[0]
    b = jnp.ones(batch_next.shape[0]) / batch_next.shape[0]

    geom = pointcloud.PointCloud(batch, batch_next, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a, b)

    solver = sinkhorn.Sinkhorn()
    out = solver(prob)

    weights = out.matrix

    return compute_relevant_couplings(batch, batch_next, time, weights)


def compute_couplings(batch: jnp.ndarray, batch_next: jnp.ndarray, time: int) -> jnp.ndarray:
    """
    Computes the couplings between particles in two consecutive batches.

    This function uses the `wasserstein_couplings` function, which leverages the POT (Python Optimal Transport) library
    to compute the optimal transport plan between two sets of particles.

    Parameters
    ----------
    batch : jnp.ndarray
        The array of particles at the current timestep with shape (n_particles, n_features).

    batch_next : jnp.ndarray
        The array of particles at the next timestep with shape (n_particles, n_features).

    time : int
        The timestep of batch_next.

    Returns
    -------
    jnp.ndarray
        An array of shape (n_relevant_couplings, 2 * n_features + 2) where each row contains:

        - Particle from `batch` (shape: (n_features,))
        - Particle from `batch_next` (shape: (n_features,))
        - Time (float)
        - Coupling weight (float)

        Only the relevant couplings, where the weight is greater than a threshold, are included.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> batch = jnp.array([[0., 0.], [1., 0.]])
    >>> batch_next = jnp.array([[0., 1.], [2., 2.]])
    >>> time = 5
    >>> compute_couplings(batch, batch_next, time)
    DeviceArray([[0. , 0. , 0. , 1. , 5. , 0.5],
                 [1. , 0. , 2. , 2. , 5. , 0.5]], dtype=float32)

    References
    ----------
    - POT library documentation: https://pythonot.github.io/

    """

    chex.assert_rank(batch, 2)
    chex.assert_rank(batch_next, 2)
    chex.assert_axis_dimension(batch, axis=1, expected=batch_next.shape[1])
    chex.assert_type([batch, batch_next], float)
    chex.assert_type(time, int)

    weights = wasserstein_couplings(batch, batch_next)

    return compute_relevant_couplings(batch, batch_next, time, weights)


def compute_relevant_couplings(batch, batch_next, time, weights):
    """
    Computes the relevant couplings between particles in two consecutive batches by filtering out the couplings with low weights.

    Parameters
    ----------
    batch : jnp.ndarray
        The array of particles at the current timestep with shape (n_particles, n_features).
    batch_next : jnp.ndarray
        The array of particles at the next timestep with shape (n_particles, n_features).
    time : int
        The timestep of batch_next.
    weights : jnp.ndarray
        The optimal transport plan matrix of shape (n_samples_x, n_samples_y).

    Returns
    -------
    jnp.ndarray
        An array of shape (n_relevant_couplings, 2 * n_features + 2) where each row contains:

        - Particle from `batch` (shape: (n_features,))
        - Particle from `batch_next` (shape: (n_features,))
        - Time (float)
        - Coupling weight (float)

        Only the relevant couplings, where the weight is greater than a threshold, are included.
    """
    # Create particle indices
    idx_t = jnp.arange(batch.shape[0])
    idx_t_next = jnp.arange(batch_next.shape[0])
    idx_t, idx_t_next \
        = jnp.meshgrid(idx_t, idx_t_next, indexing='ij')
    x = batch[idx_t.flatten()]
    y = batch_next[idx_t_next.flatten()]

    # Stack the columns so to have particle_x, particle_y, coupling_weight on each row
    couplings = jnp.column_stack((x, y, jnp.full_like(weights.flatten(), time), weights.flatten()))


    # Pick top couplings (~transport map)
    min_probability = 1 / (10 * max(batch.shape[0], batch_next.shape[0]))
    relevant_couplings = []
    while len(relevant_couplings) == 0:
        relevant_couplings = couplings[couplings[:, -1] > min_probability]
        min_probability /= 2

    return relevant_couplings