import jax

def network_grad(network, params):
    return jax.vmap(lambda v: jax.grad(network.apply, argnums=1)({'params': params}, v))

def network_grad_time(network, params):
    def grad_fn(v):
        partial_v = v[:-1]
        def loss_fn(partial_input):
            full_input = jax.numpy.concatenate([partial_input, v[-1:]], axis=-1)
            return network.apply({'params': params}, full_input)
        return jax.grad(loss_fn)(partial_v)
    return jax.vmap(grad_fn, in_axes=0)

def count_parameters(model):
    return sum(map(lambda x: x.size, jax.tree_flatten(model)[0]))