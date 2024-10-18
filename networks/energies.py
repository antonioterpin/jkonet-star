"""
Models for energy functions.
"""


import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence


class MLP(nn.Module):
    """
    Simple energy model.
    
    Source: https://github.com/bunnech/jkonet
    """

    dim_hidden: Sequence[int]
    act_fn: Callable = nn.softplus

    def setup(self):
        num_hidden = len(self.dim_hidden)

        layers = list()
        for i in range(num_hidden):
            layers.append(nn.Dense(features=self.dim_hidden[i]))
        layers.append(nn.Dense(features=1))
        self.layers = layers

    @nn.compact
    def __call__(self, x, s=True):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))
        y = self.layers[-1](x)
        if s:
            return jnp.sum(y)
        else:
            return y