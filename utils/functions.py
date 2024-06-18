"""
A collection of functions for energy landscapes.
These functions are not vectorized on purpose so that we can use jax.grad.
To get automatic vectorization use jax.vmap.
"""

import jax.numpy as jnp

def styblinski_tang(v: jnp.ndarray) -> jnp.ndarray:
    u = jnp.square(v)
    return 0.5 * jnp.sum(jnp.square(u) - 16 * u + 5 * v)

def holder_table(v: jnp.ndarray) -> jnp.ndarray:
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return 10 * jnp.abs(jnp.sin(v1) * jnp.cos(v2) * jnp.exp(jnp.abs(1 - jnp.sqrt(jnp.sum(jnp.square(v)))/jnp.pi)))

def cross_in_tray(v: jnp.ndarray) -> jnp.ndarray:
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return -2 * (jnp.abs(jnp.sin(v1) * jnp.sin(v2) * jnp.exp(jnp.abs(10 - jnp.sqrt(jnp.sum(jnp.square(v)))/jnp.pi))) + 1)**0.1

def oakley_ohagan(v: jnp.ndarray) -> jnp.ndarray:
    return 5 * jnp.sum(jnp.sin(v) + jnp.cos(v) + jnp.square(v) + v)

def moon(v: jnp.ndarray) -> jnp.ndarray:
    interaction_matrix = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-2.08, 1.42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.11, 2.18, -1.70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.76, 0.58, 0.84, 1.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.57, -1.21, 1.20, -0.49, -3.23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.72, -7.15, -2.35, 1.74, 2.75, -1.10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.47, -1.29, -0.16, 1.29, -1.40, 2.34, 0.21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.39, -0.19, -0.35, 0.24, -3.90, -0.03, -4.16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.40, -2.75, -5.93, -4.73, -0.70, -0.80, -0.37, 0.26, -1.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.09, -1.16, -1.15, 3.27, -0.17, 0.13, -1.27, -0.30, 0.77, 3.06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.7, -1.09, 1.89, 1.87, -3.38, -3.97, 2.78, -2.69, 1.09, 2.46, 3.34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.27, 0.89, -3.47, 1.42, -1.87, 1.99, 1.37, -2.56, -1.15, 5.80, 2.36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.03, -0.16, -0.07, -0.96, -0.17, 0.45, -2.75, 28.99, -1.09, -5.15, -1.77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.07, 4.43, 0.60, -0.91, 1.56, 1.77, -3.15, -2.13, -2.74, -2.05, -3.16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.23, 1.65, -1.09, 2.06, 2.40, -0.50, 1.86, 1.36, 1.59, 3.17, 1.89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.46, -1.25, -3.23, 2.89, -1.70, 1.86, 0.12, 1.45, .41, 3.40, 2.20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.31, -1.35, 0.44, 0.25, 0.32, 0.02, -0.74, 3.09, 0.48, -0.49, -0.71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-2.94, 1.15, 1.24, 1.97, 2.11, -2.08, 1.06, -1.73, 2.16, -6.71, -3.78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.63, -19.71, 2.13, 3.04, -0.20, 1.78, -3.76, -1.66, 0.34, -0.74, 0.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.07, 23.72, -0.71, 2.00, 1.39, 1.76, -0.43, -3.94, 4.17, 2.78, 1.40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.44, 1.42, 1.64, 1.64, -2.01, 1.30, 1.25, -2.56, 0.73, -0.41, -0.59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    v = v / 4
    d = v.shape[0]
    while d < 20:
        v = jnp.concatenate([v, v ** 2], axis=0)
        d = v.shape[0]
    v = jnp.concatenate([jnp.ones(1), v[:20]], axis=0)
    return jnp.clip(jnp.dot(v, jnp.dot(interaction_matrix, v)), -100, 100)

def ishigami(v: jnp.ndarray) -> jnp.ndarray:
    d = v.shape[0]
    v0 = jnp.mean(v[:d//2])
    v1 = jnp.mean(v[d//2:])
    v2 = (v0 + v1) / 2
    return jnp.sin(v0) + 7 * jnp.sin(v1) ** 2 + 0.1 * v2 ** 4 * jnp.sin(v0)

def friedman(v: jnp.ndarray) -> jnp.ndarray:
    v = 2 * (v - 7)
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:]) / 2
    v3 = v1 * jnp.sin(v2)
    v4 = v1 * jnp.cos(v2)
    v5 = v2 * jnp.sin(v1)
    return (10 * jnp.sin(jnp.pi * v1 * v2) + 20 * (v3 - 0.5) ** 2 + 10 * (v4 - 1) ** 2 + 0.1 * v5) / 100

def sphere(v: jnp.ndarray) -> jnp.ndarray:
    return -10 * jnp.sum(jnp.square(v))

def bohachevsky(v: jnp.ndarray) -> jnp.ndarray:
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return 10 * (jnp.square(v1) + 2 * jnp.square(v2) - 0.3 * jnp.cos(3 * jnp.pi * v1) - 0.4 * jnp.cos(4 * jnp.pi * v2))

def three_hump_camel(v: jnp.ndarray) -> jnp.ndarray:
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return 2 * jnp.square(v1) - 1.05 * jnp.power(v1, 4) + jnp.power(v1, 6) / 6 + v1 * v2 + jnp.square(v2)

def beale(v: jnp.ndarray) -> jnp.ndarray:
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return (jnp.square(1.5 - v1 + v1 * v2) + jnp.square(2.25 - v1 + v1 * jnp.square(v2)) + jnp.square(2.625 - v1 + v1 * jnp.power(v2, 3))) / 100

def double_exp(v: jnp.ndarray) -> jnp.ndarray:
    s = 20
    d = 3
    return 200 * (jnp.exp(-jnp.sum(jnp.square(v - d))/s) + jnp.exp(-jnp.sum(jnp.square(v + d))/s))

def relu(v: jnp.ndarray) -> jnp.ndarray:
    r = -50 * jnp.clip(v, a_min=0)
    if r.ndim > 0:
        return jnp.sum(r)
    return r

def rotational(v: jnp.ndarray) -> jnp.ndarray:
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    theta = jnp.arctan2(v2 + 5, v1 + 5)
    return 10 * relu(theta + jnp.pi)

def flat(v: jnp.ndarray) -> jnp.ndarray:
    return 0.


potentials_all = {
    'double_exp': double_exp,
    'rotational': rotational,
    'relu': relu,
    'flat': flat,
    'beale': beale,
    'friedman': friedman,
    'moon': moon,
    'ishigami': ishigami,
    'three_hump_camel': three_hump_camel,
    'bohachevsky': bohachevsky,
    'holder_table': holder_table,
    'cross_in_tray': cross_in_tray,
    'oakley_ohagan': oakley_ohagan,
    'sphere': sphere,
    'styblinski_tang': styblinski_tang
}

interactions_all = potentials_all