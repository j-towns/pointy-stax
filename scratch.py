from functools import reduce
import jax.numpy as np
import jax.linear_util as lu
from jax.util import unzip2, safe_zip, safe_map, partial
from jax.abstract_arrays import ShapedArray
from jax.experimental import stax
from jax.interpreters import partial_eval as pe
from jax.interpreters.batching import get_aval
from jax.api_util import (
    wraps, pytree_to_jaxtupletree, pytree_fun_to_jaxtupletree_fun)
import jax.core as jc
from jax import random
from jax import lax


zip = safe_zip
map = safe_map


RNG = [random.PRNGKey(0)]

def sample_rng():
    key1, key2 = random.split(RNG.pop())
    RNG.append(key1)
    return key2

def set_rng(key):
    RNG.pop()
    RNG.append(key)

def merge_params(params):
    if len(params) > 0:
        p = params[0]
        for param in params[1:]:
            p.update(param)
        return p
    else:
        return {}

class StaxLayer(jc.Primitive):
    def __init__(self, name, init_fun, apply_fun):
        self.init_fun = init_fun
        self.apply_fun = apply_fun
        super(StaxLayer, self).__init__(name)
        def layer_abstract_eval(*avals):
            akey = ShapedArray((2,), 'uint32')
            def init_and_apply(key, *inputs):
                params = init_fun(key, *inputs)
                return apply_fun(params, *inputs)
            return pe.abstract_eval_fun(init_and_apply, akey, *avals)
        self.def_abstract_eval(layer_abstract_eval)

class InitTracer(jc.Tracer):
    __slots__ = ['val', 'net_params']

    def __init__(self, trace, net_params, val):
        self.trace = trace
        self.val = val
        self.net_params = net_params

    @property
    def aval(self):
        return jc.get_aval(self.val)

    def unpack(self):
        return tuple(self.val)

    def full_lower(self):
        return self

class InitTrace(jc.Trace):
    def pure(self, val):
        return InitTracer(self, {}, val)

    def lift(self, val):
        return InitTracer(self, {}, val)

    def sublift(self, val):
        return InitTracer(self, {}, val.val)

    def process_primitive(self, primitive, tracers, params):
        vals_in, net_params = zip(*[(t.val, t.net_params) for t in tracers])
        net_params = merge_params(net_params)
        if isinstance(primitive, StaxLayer):
            apply_fun = primitive.apply_fun
            if primitive.name in net_params:
                layer_params = net_params[primitive.name]
            else:
                init_fun = primitive.init_fun
                layer_params = init_fun(sample_rng(), *vals_in)
                net_params[primitive.name] = layer_params
            return InitTracer(
                self, net_params, apply_fun(layer_params, *vals_in))
        else:
            return InitTracer(
                self, net_params, primitive.bind(*vals_in, **params))

@lu.transformation_with_aux
def init_transform(rng, inputs):
    set_rng(rng)
    with jc.new_master(InitTrace) as master:
        trace = InitTrace(master, jc.cur_sublevel())
        ans = yield map(partial(InitTracer, trace, {}), inputs), {}
        out_tracer = trace.full_raise(ans)
        out_val, net_params = out_tracer.val, out_tracer.net_params
        del master, out_tracer
    yield out_val, net_params

def init_fun(net_fun, rng, *example_inputs):
    net_fun = lu.wrap_init(net_fun)
    net_fun, net_params = init_transform(net_fun, rng)
    net_fun.call_wrapped(example_inputs)
    return net_params()

class ApplyTracer(jc.Tracer):
    __slots__ = ['val', 'net_params']

    def __init__(self, trace, net_params, val):
        self.trace = trace
        self.val = val
        self.net_params = net_params

    @property
    def aval(self):
        return jc.get_aval(self.val)

    def unpack(self):
        return tuple(self.val)

    def full_lower(self):
        return self

class ApplyTrace(jc.Trace):
    def pure(self, val):
        return ApplyTracer(self, {}, val)

    def lift(self, val):
        return ApplyTracer(self, {}, val)

    def sublift(self, val):
        return ApplyTracer(self, {}, val.val)

    def process_primitive(self, primitive, tracers, params):
        vals_in, net_params = zip(*[(t.val, t.net_params) for t in tracers])
        net_params = merge_params(net_params)
        if isinstance(primitive, StaxLayer):
            apply_fun = primitive.apply_fun
            layer_params = net_params[primitive.name]
            return ApplyTracer(
                self, net_params, apply_fun(layer_params, *vals_in))
        else:
            return ApplyTracer(
                self, net_params, primitive.bind(*vals_in, **params))

@lu.transformation
def apply_transform(net_params, inputs):
    with jc.new_master(ApplyTrace) as master:
        trace = ApplyTrace(master, jc.cur_sublevel())
        ans = yield map(partial(ApplyTracer, trace, net_params), inputs), {}
        out_tracer = trace.full_raise(ans)
        out_val = out_tracer.val
        del master, out_tracer
    yield out_val

def apply_fun(net_fun, params, *inputs):
    return apply_transform(lu.wrap_init(net_fun), params).call_wrapped(inputs)

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
    return StaxLayer(name, init_fun, apply_fun).bind


# DENSE DEMO
my_dense_layer = Dense('my_dense_layer', out_dim=2)

example_inputs = np.array([1., 1.])  # example_inputs.shape == (2,)

def dense_net_fun(inputs):
    return my_dense_layer(2 * my_dense_layer(inputs))

dense_params = init_fun(dense_net_fun, random.PRNGKey(0), example_inputs)
dense_out = apply_fun(dense_net_fun, dense_params, example_inputs)

# RNN DEMO
def rnn_cell(hidden, inputs):
    hidden = my_dense_layer(inputs) * my_dense_layer(hidden)
    return hidden, hidden

hidden_init = example_inputs
def rnn(inputs):
    return lax.scan(rnn_cell, hidden_init, inputs)

rnn_example_inputs = np.stack([example_inputs, example_inputs, example_inputs])

rnn_params = init_fun(rnn, random.PRNGKey(0), rnn_example_inputs)
