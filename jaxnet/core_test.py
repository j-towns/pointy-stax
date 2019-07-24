from jax import random, vmap, jit
import jax.numpy as np
import core


RNG = random.PRNGKey(0)

def Layer(name):
    out_dim = 3
    def init_fun(rng, example_input):
        input_shape = example_input.shape
        k1, k2 = random.split(rng)
        W, b = (random.normal(k1, (out_dim, input_shape[-1])),
                random.normal(k2, (out_dim,)))
        return W, b
    def apply_fun(params, inputs):
        W, b = params
        return np.dot(W, inputs) + b
    return core.Layer(name, init_fun, apply_fun).bind

layer = Layer("Test layer")

def test_init_and_apply():
    example_inputs = random.normal(RNG, (2,))
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_inputs)
    out = core.apply_fun(net_fun, params, example_inputs)
    assert out.shape == (3,)

def test_batch_apply():
    example_input = random.normal(RNG, (2,))
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_input)
    def apply(inputs):
        return core.apply_fun(net_fun, params, inputs)
    example_input_batch = np.stack(4 * [example_input])
    out = vmap(apply)(example_input_batch)
    assert out.shape == (4, 3)

def test_apply_batch():
    example_input = random.normal(RNG, (2,))
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_input)
    example_input_batch = np.stack(4 * [example_input])
    out = core.apply_fun(vmap(net_fun), params, example_input_batch)
    assert out.shape == (4, 3)

def test_jit_apply():
    example_inputs = random.normal(RNG, (2,))
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_inputs)

    @jit
    def jittable(params, example_inputs):
        return core.apply_fun(net_fun, params, example_inputs)

    out = jittable(params, example_inputs)
    assert out.shape == (3,)
    out_ = jittable(params, example_inputs)
    assert out_.shape == (3,)

def test_apply_jit():
    example_inputs = random.normal(RNG, (2,))
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_inputs)
    net_fun_jitted = jit(net_fun)
    out = core.apply_fun(net_fun_jitted, params, example_inputs)
    assert out.shape == (3,)
