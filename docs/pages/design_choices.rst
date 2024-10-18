Design choices and tips for effectively deploying JKOnet\* ☝️
================================================================

This section contains a work-in-progress collection of design strategies and tips for effectively deploying JKOnet\*.

Mixture of Gaussians for Density Estimation
-------------------------------------------

When considering the internal energy term of the energy functional :math:`J` (i.e., when :math:`\theta_3 \neq 0`), we need to estimate the density :math:`\rho_t` and its gradient :math:`\nabla \rho_t` from the empirical probability measures :math:`\mu_t`. To estimate :math:`\rho_t`, we employ a mixture of 10 Gaussians. This method represents :math:`\rho_t` as a weighted sum of multiple Gaussian distributions, each with its own mean and variance, allowing for a flexible and accurate approximation of complex density functions. The gradient :math:`\nabla \rho_t` is then computed by automatic differentiation (using :code:`JAX`) but it could be computed analytically.

Tips and possible improvements:

- The number of Gaussians in the mixture can be adjusted based on the complexity of the density function to be approximated (see information on the data generation page).
- The parametrization of the density may be adapted to the specific problem at hand, e.g., by using a different type of distribution.
- It may be worth exploring the use of more sophisticated density estimation techniques, such as kernel density estimation or neural density estimation.


Precomputation of Couplings
----------------------------

Precomputing couplings (our approach) speeds up the optimization process by avoiding redundant computations.

If the number of couplings to store is large, consider computing them on the fly during training. This approach may be more memory-efficient but increases the computational cost per iteration.

If the number of particles is large, computing couplings may become computationally expensive. If this is the case:

- Consider batching the particles, computing couplings for each batch.
- Consider exploring other algorithms for computing couplings, such as the Sinkhorn algorithm.


Linear vs non-linear parametrization
--------------------------------------

Linear parametrization provides several benefits, including more predictable performance and often better results when the features are carefully selected. However, it tends to scale poorly with dimensionality and requires careful selection of representative features. In contrast, non-linear parametrization, such as in neural networks, automates feature selection and tends to perform better in high-dimensional settings.
