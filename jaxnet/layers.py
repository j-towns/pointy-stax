from jax import random
from jax.experimental import stax

import core


def Dense(name, out_dim, W_init=stax.glorot(), b_init=stax.randn()):
    """Layer constructor function for a dense (fully-connected) layer."""
    def init_fun(rng, example_input):
        input_shape = example_input.shape
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (out_dim, input_shape[-1])), b_init(k2, (out_dim,))
        return W, b
    def apply_fun(params, inputs):
        W, b = params
        return np.dot(W, inputs) + b
    return core.Layer(name, init_fun, apply_fun).bind
