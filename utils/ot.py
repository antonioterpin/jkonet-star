import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
import ot
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

def wasserstein_couplings(xs, ys):
    """
    Computes transport between xs and ys.
    """
    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    M = ot.dist(xs, ys)

    return ot.emd(a, b, M)

def wasserstein_loss(xs, ys):
    """
    Computes transport between xs and ys.
    """
    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    M = ot.dist(xs, ys)

    return ot.emd2(a, b, M)

@jax.jit
def sinkhorn_loss(xs, ys, epsilon=1):
    """
    Computes transport between xs and ys.
    """
    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    geom = pointcloud.PointCloud(xs, ys, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a, b)

    solver = sinkhorn.Sinkhorn()
    out = solver(prob)

    return out.reg_ot_cost


def compute_couplings(batch, batch_next):
    """
    Computes transport between batch and batch_next.

    Returns:
    - for each particle in batch, the particle in batch_next it is coupled with together with the coupling weight
    """
    weights = wasserstein_couplings(batch, batch_next)

    # Create particle indices
    idx_t = jnp.arange(batch.shape[0])
    idx_t_next = jnp.arange(batch_next.shape[0])
    idx_t, idx_t_next \
        = jnp.meshgrid(idx_t, idx_t_next, indexing='ij')
    x = batch[idx_t.flatten()]
    y = batch_next[idx_t_next.flatten()]

    # Stack the columns so to have particle_x, particle_y, coupling_weight on each row
    couplings = jnp.column_stack((x, y, weights.flatten()))

    # Pick top couplings (~transport map)
    couplings = couplings[(-couplings[:, -1]).argsort()][0:batch.shape[0]]

    return couplings