import jax

def network_grad(network, params):
    return jax.vmap(lambda v: jax.grad(network.apply, argnums=1)({'params': params}, v))

def count_parameters(model):
    return sum(map(lambda x: x.size, jax.tree_flatten(model)[0]))