# source: https://github.com/bunnech/jkonet
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Sequence, Tuple

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


class Dense(nn.Module):
    """A linear transformation applied over the last dimension of the input.
    Attributes:
    dim_hidden: the number of output dim_hidden.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    """
    dim_hidden: int
    beta: float = 1.0
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable[
        [PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        """Applies a linear transformation to inputs along the last dimension.
        Args:
          inputs: The nd-array to be transformed.
        Returns:
          The transformed input.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param('kernel',
                            self.kernel_init,
                            (inputs.shape[-1], self.dim_hidden))
        scaled_kernel = self.beta * kernel
        kernel = jnp.asarray(1 / self.beta * nn.softplus(scaled_kernel),
                             self.dtype)
        y = jax.lax.dot_general(inputs, kernel,
                                (((inputs.ndim - 1,), (0,)), ((), ())),
                                precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.dim_hidden,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y


class ICNN(nn.Module):
    """Input convex neural network."""

    dim_hidden: Sequence[int]
    init_std: float = 0.1
    init_fn: str = 'normal'
    act_fn: Callable = nn.leaky_relu
    pos_weights: bool = True

    def setup(self):
        num_hidden = len(self.dim_hidden)

        w_zs = list()

        if self.pos_weights:
            w_z = Dense
        else:
            w_z = nn.Dense

        if self.init_fn == 'uniform':
            init_fn = jax.nn.initializers.uniform
        else:
            init_fn = jax.nn.initializers.normal

        for i in range(1, num_hidden):
            w_zs.append(w_z(self.dim_hidden[i],
                            kernel_init=init_fn(self.init_std),
                            use_bias=False))
        w_zs.append(w_z(1, kernel_init=init_fn(
                    self.init_std), use_bias=False))
        self.w_zs = w_zs

        w_xs = list()
        for i in range(num_hidden):
            w_xs.append(nn.Dense(self.dim_hidden[i],
                                 kernel_init=init_fn(self.init_std),
                                 use_bias=True))
        w_xs.append(nn.Dense(1,
                             kernel_init=init_fn(self.init_std),
                             use_bias=True))
        self.w_xs = w_xs

    @nn.compact
    def __call__(self, x):
        z = self.act_fn(self.w_xs[0](x))
        z = jnp.multiply(z, z)

        for w_z, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
            z = self.act_fn(jnp.add(w_z(z), Wx(x)))
        y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))

        return jnp.squeeze(y)
