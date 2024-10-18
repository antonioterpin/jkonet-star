"""
This module provides a collection of energy landscape functions commonly used for optimization, testing, and benchmarking purposes. These functions are intentionally not vectorized to enable the use of ``jax.grad`` for automatic differentiation. For automatic vectorization, you can use ``jax.vmap``.

The functions in this module represent a variety of optimization landscapes, including convex, non-convex, and complex synthetic functions. They are commonly used in sensitivity analysis, regression tasks, and testing optimization algorithms.

Available Functions:
---------------------
- ``styblinski_tang``: A non-convex function used to test optimization algorithms.
- ``holder_table``: A complex, non-convex optimization function.
- ``zigzag_ridge``: Another non-convex function, used for benchmarking optimization algorithms.
- ``oakley_ohagan``: A synthetic function used for testing.
- ``watershed``: A complex function that uses an interaction matrix for polynomial expansions.
- ``ishigami``: A function used in sensitivity analysis.
- ``friedman``: A function used in regression and sensitivity analysis.
- ``sphere``: A simple convex function for optimization testing.
- ``bohachevsky``: A non-convex function with trigonometric terms.
- ``flowers``: A non-convex optimization function.
- ``wavy_plateau``: A well-known non-convex function used for optimization testing.
- ``double_exp``: A double exponential function used in optimization problems.
- ``relu``: A rectified linear unit (ReLU) function.
- ``rotational``: A trigonometric-based optimization function.
- ``flat``: A trivial function that returns zero, useful for testing.

Example Usage:
--------------
To use any of the provided functions, pass a ``jax.numpy`` array as input:

.. code-block:: python

    import jax.numpy as jnp
    from module_name import potentials_all

    v = jnp.array([1.0, 2.0, 3.0])
    result = potentials_all['styblinski_tang'](v)

You can also compute the gradient of these functions using `jax.grad`:

.. code-block:: python

    from jax import grad
    gradient = grad(potentials_all['styblinski_tang'])(v)

Note:
-----
For vectorized operations, you can use `jax.vmap` over the provided functions.
"""

import jax.numpy as jnp

def styblinski_tang(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Styblinski-Tang function.

    .. math::
        f(v) = 0.5 \sum_{i=1}^{d} (v_i^4 - 16v_i^2 + 5v_i)

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Styblinski-Tang function.
    """
    u = jnp.square(v)
    return 0.5 * jnp.sum(jnp.square(u) - 16 * u + 5 * v)

def holder_table(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Holder Table function.

    .. math::
        f(v) = -\left|\sin(v_1)\cos(v_2)\exp\left(\left|1 - \frac{\sqrt{v_1^2 + v_2^2}}{\pi}\right|\right)\right|

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Holder Table function.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return 10 * jnp.abs(jnp.sin(v1) * jnp.cos(v2) * jnp.exp(jnp.abs(1 - jnp.sqrt(jnp.sum(jnp.square(v)))/jnp.pi)))

def zigzag_ridge(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Zigzag Ridge function.

    .. math::
        f(v) = \sum_{i=1}^{d-1} \left[ |v_i - v_{i+1}|^2 + \cos(1.25 \cdot v_i) \cdot (v_i + v_{i+1}) + v_i^2 \cdot v_{i+1} \right]

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Zigzag Ridge function.
    """
    return jnp.sum(
        jnp.abs(v[:-1] - v[1:]) ** 2 + jnp.cos(v[:-1]) * (v[:-1] + v[1:]) + v[:-1] ** 2 * v[1:]
    )

def oakley_ohagan(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Oakley-Ohagan function.

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Oakley-Ohagan function.

        .. math::

            f(v) = 5 \sum_{i=1}^{d} (\sin(v_i) + \cos(v_i) + v_i^2 + v_i)
    """
    return 5 * jnp.sum(jnp.sin(v) + jnp.cos(v) + jnp.square(v) + v)

def watershed(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Watershed function.

    The function is defined as:

    .. math::
        f(v) = \frac{1}{10} \sum_{i=1}^{d-1} \left( v_i + v_i^2 \cdot (v_{i+1} + 4) \right)

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Watershed function.
    """
    return jnp.sum(
        v[:-1] + v[:-1] ** 2 * (v[1:] + 4)
    ) / 10

def ishigami(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Ishigami function.

    .. math::
        f(v) = \sin(z_1) + 7 \sin(z_2)^2 + 0.1 \left(\frac{z_1 + z_2}{2}\right)^4 \sin(z_1)

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Ishigami function.
    """

    d = v.shape[0]
    v0 = jnp.mean(v[:d//2])
    v1 = jnp.mean(v[d//2:])
    v2 = (v0 + v1) / 2
    return jnp.sin(v0) + 7 * jnp.sin(v1) ** 2 + 0.1 * v2 ** 4 * jnp.sin(v0)

def friedman(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Friedman function.

    .. math::
        f(v) = \frac{1}{100}\biggl(10\sin\left(2\pi(z_1 - 7)(z_2 - 7)\right) + 
        20\left(2(z_1 - 7)\sin(z_2 - 7)- \frac{1}{2}\right)^2 \\\\ +
        10\left(2(z_1 - 7)\cos(z_2 - 7) - 1\right)^2 + \frac{1}{10}(z_2 - 7)\sin(2(z_1 - 7))\biggr)



    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Friedman function, scaled down by a factor of 100.
    """
    v = 2 * (v - 7)
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:]) / 2
    v3 = v1 * jnp.sin(v2)
    v4 = v1 * jnp.cos(v2)
    v5 = v2 * jnp.sin(v1)
    return (10 * jnp.sin(jnp.pi * v1 * v2) + 20 * (v3 - 0.5) ** 2 + 10 * (v4 - 1) ** 2 + 0.1 * v5) / 100

def sphere(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Sphere function.

    .. math::
        f(v) = -10||x||^2

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Sphere function.
    """
    return -10 * jnp.sum(jnp.square(v))

def bohachevsky(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Bohachevsky function.

    .. math::
        f(v) = v_1^2 + 2v_2^2 - 0.3 \cos(3 \pi v_1) - 0.4 \cos(4 \pi v_2) + 0.7


    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Bohachevsky function.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return 10 * (jnp.square(v1) + 2 * jnp.square(v2) - 0.3 * jnp.cos(3 * jnp.pi * v1) - 0.4 * jnp.cos(4 * jnp.pi * v2))

def flowers(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Flowers function.

    The function is defined as:

    .. math::
        f(v) = \sum_{i=1}^{d} \left[ v_i + 2 \cdot \sin(|v_i|^{1.2}) \right]

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Flowers function.
    """
    return jnp.sum(
        v + 2 * jnp.sin(jnp.abs(v) ** 1.2)
    )

def wavy_plateau(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Wavy Plateau function.

    The function is defined as:

    .. math::
        f(v) = \sum_{i=1}^{d} \left[\cos(5 \pi v_i) + 0.5 \cdot v_i^4 - 3 \cdot v_i^2 + 1 \right]

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Wavy Plateau function.
    """
    return jnp.sum(
        jnp.cos(jnp.pi * v) + 0.5 * v**4 - 3 * v**2 + 1
    )

def double_exp(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Double Exponential function.

    .. math::
        f(v) = 200\exp\left(-\frac{||v - m\mathbf{1}||^2}{\sigma}\right) + \exp\left(-\frac{||v + m\mathbf{1}||}{s}\right)

    where :math:`d = 3` and :math:`s = 20`.

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Double Exponential function.
    """
    s = 20
    d = 3
    return 200 * (jnp.exp(-jnp.sum(jnp.square(v - d))/s) + jnp.exp(-jnp.sum(jnp.square(v + d))/s))

def relu(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the ReLU (Rectified Linear Unit) function.

    .. math::
        f(v) = \max(0, v)

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the ReLU function.
    """
    r = -50 * jnp.clip(v, a_min=0)
    if r.ndim > 0:
        return jnp.sum(r)
    return r

def rotational(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Rotational function.

    .. math::
        f(v) = 10 \cdot \text{ReLU}(\theta + \pi)

    where :math:`\theta = \arctan\left(\frac{v_2 + 5}{v_1 + 5}\right)`.

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Rotational function.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    theta = jnp.arctan2(v2 + 5, v1 + 5)
    return 10 * relu(theta + jnp.pi)

def flat(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Flat function.

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Flat function (always 0).
    """
    return 0.


potentials_all = {
    'double_exp': double_exp,
    'rotational': rotational,
    'relu': relu,
    'flat': flat,
    'wavy_plateau': wavy_plateau,
    'friedman': friedman,
    'watershed': watershed,
    'ishigami': ishigami,
    'flowers': flowers,
    'bohachevsky': bohachevsky,
    'holder_table': holder_table,
    'zigzag_ridge': zigzag_ridge,
    'oakley_ohagan': oakley_ohagan,
    'sphere': sphere,
    'styblinski_tang': styblinski_tang
}

interactions_all = potentials_all