from jax import grad, vmap, numpy as jnp

# RBFs
def rbf_linear(x, c):
    return - jnp.linalg.norm((x - c))

def rbf_thin_plate_spline(x, c):
    r = jnp.linalg.norm(x - c)
    return r ** 2 * jnp.log(r + 1e-6)

def rbf_cubic(x, c):
    return jnp.sum((x - c) ** 3)

def rbf_quintic(x, c):
    return -jnp.sum((x - c) ** 5)

def rbf_multiquadric(x, c):
    return -jnp.sqrt(jnp.sum((x - c) ** 2) + 1)

def rbf_inverse_multiquadric(x, c):
    return 1 / jnp.sqrt(jnp.sum((x - c) ** 2) + 1)

def rbf_inverse_quadratic(x, c):
    return 1 / (jnp.sum((x - c) ** 2) + 1)

def const(x, c):
    return 1
    
rbfs = {
    'linear': rbf_linear,
    'thin_plate_spline': rbf_thin_plate_spline,
    'cubic': rbf_cubic,
    'quintic': rbf_quintic,
    'multiquadric': rbf_multiquadric,
    'inverse_multiquadric': rbf_inverse_multiquadric,
    'inverse_quadratic': rbf_inverse_quadratic,
    'const': const
}