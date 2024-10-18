r"""
Radial Basis Function (RBF) Module

This module provides implementations of various types of radial basis functions (RBFs),
which are commonly used in interpolation, machine learning, and numerical approximation.

Radial basis functions take two inputs:
    - A data point `x` (a jax.numpy array).
    - A center `c` (a jax.numpy array).
    
Each RBF computes a scalar value based on the distance (or other non-linear transformation)
between the input `x` and the center `c`.

Available Functions:
---------------------
- ``rbf_linear``: Computes the linear RBF, which is the negative Euclidean distance between `x` and `c`.
- ``rbf_thin_plate_spline``: Computes the thin plate spline RBF, which is proportional to the squared distance
  multiplied by the logarithm of the distance.
- ``rbf_cubic``: Computes the cubic RBF, which sums the cube of differences between `x` and `c`.
- ``rbf_quintic``: Computes the quintic RBF, which sums the fifth power of the differences between `x` and `c`.
- ``rbf_multiquadric``: Computes the multiquadric RBF, which is the negative square root of the sum of squared
  differences plus a constant.
- ``rbf_inverse_multiquadric``: Computes the inverse of the multiquadric RBF.
- ``rbf_inverse_quadratic``: Computes the inverse quadratic RBF.
- ``const``: A constant function that always returns 1 regardless of the inputs.

The `rbfs` dictionary provides a convenient way to access these RBF functions by name.

Usage Example:
--------------
To compute an RBF between an input data point and a center, you can use one of the provided RBF functions:

.. code-block:: python

    import jax.numpy as jnp
    from module_name import rbfs

    x = jnp.array([1.0, 2.0])
    c = jnp.array([0.5, 1.5])
    rbf_value = rbfs['linear'](x, c)

This will compute the linear RBF between `x` and `c`.
"""


from jax import numpy as jnp

def rbf_linear(x, c):
    r"""
    Computes the linear radial basis function (RBF).

    This RBF is simply the negative Euclidean distance between the input ``x``
    and the center ``c``.

    .. math::

        \mathrm{RBF}(x, c) = -\|x - c\|

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the linear RBF.
    """
    return - jnp.linalg.norm((x - c))

def rbf_thin_plate_spline(x, c):
    r"""
    Computes the thin plate spline radial basis function (RBF).

    .. math::

        \mathrm{RBF}(x, c) = \|x - c\|^2 \log(\|x - c\| + \epsilon)

    where :math:`\epsilon` is a small constant to avoid numerical issues when
    :math:`\|x - c\|` is close to zero.

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the thin plate spline RBF.
    """
    r = jnp.linalg.norm(x - c)
    return r ** 2 * jnp.log(r + 1e-6)

def rbf_cubic(x, c):
    r"""
    Computes the cubic radial basis function (RBF).

    .. math::

        \mathrm{RBF}(x, c) = \sum_{i} (x_i - c_i)^3

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the cubic RBF.
    """
    return jnp.sum((x - c) ** 3)

def rbf_quintic(x, c):
    r"""
    Computes the quintic radial basis function (RBF).

    .. math::

        \mathrm{RBF}(x, c) = -\sum_{i} (x_i - c_i)^5

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the quintic RBF.
    """
    return -jnp.sum((x - c) ** 5)

def rbf_multiquadric(x, c):
    r"""
    Computes the multiquadric radial basis function (RBF).

    .. math::

        \mathrm{RBF}(x, c) = -\sqrt{\sum_{i} (x_i - c_i)^2 + 1}

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the multiquadric RBF.
    """
    return -jnp.sqrt(jnp.sum((x - c) ** 2) + 1)

def rbf_inverse_multiquadric(x, c):
    r"""
    Computes the inverse multiquadric radial basis function (RBF).

    .. math::

        \mathrm{RBF}(x, c) = \left( \sqrt{\sum_{i} (x_i - c_i)^2 + 1} \right)^{-1}

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the inverse multiquadric RBF.
    """
    return 1 / jnp.sqrt(jnp.sum((x - c) ** 2) + 1)

def rbf_inverse_quadratic(x, c):
    r"""
    Computes the inverse quadratic radial basis function (RBF).

    .. math::

        \mathrm{RBF}(x, c) = \left(\sum_{i} (x_i - c_i)^2 + 1\right)^{-1}

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the inverse quadratic RBF.
    """
    return 1 / (jnp.sum((x - c) ** 2) + 1)

def const(x, c):
    r"""
    Computes the constant function.

    This function always returns 1, regardless of the input ``x`` or the center ``c``.

    .. math::

        \mathrm{const}(x, c) = 1

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center (unused).

    Returns:
        jnp.ndarray: The constant value 1.
    """
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