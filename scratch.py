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


def merge_params(params):
    if len(params) > 0:
        p = params[0]
        for param in params[1:]:
            p.update(param)
        return p
    else:
        return {}

class Layer(jc.Primitive):
    def __init__(self, name, init_fun, apply_fun):
        self.init_fun = init_fun
        self.apply_fun = apply_fun
        super(Layer, self).__init__(name)
        def layer_abstract_eval(*avals):
            akey = ShapedArray((2,), 'uint32')
            def init_and_apply(key, *inputs):
                params = init_fun(key, *inputs)
                return apply_fun(params, *inputs)
            return pe.abstract_eval_fun(init_and_apply, akey, *avals)
        self.def_abstract_eval(layer_abstract_eval)

def init_interpreter(rng, jaxpr, consts, freevar_vals, net_params, *args):
    def read(v):
        if type(v) is jc.Literal:
            return v.val
        else:
            return env[v]

    def write(v, val):
        env[v] = val

    env = {}
    write(jc.unitvar, jc.unit)
    jc.pat_fmap(write, jaxpr.constvars, consts)
    jc.pat_fmap(write, jaxpr.invars, args)
    jc.pat_fmap(write, jaxpr.freevars, freevar_vals)
    for eqn in jaxpr.eqns:
        rng, prim_rng = random.split(rng)
        if not eqn.restructure:
            in_vals = map(read, eqn.invars)
        else:
            in_vals = [pack(map(read, invars)) if type(invars) is tuple
                       else read(invars) for invars in eqn.invars]
        if eqn.bound_subjaxprs:
            subjaxprs, sub_consts, sub_freevar_vals = unzip3([
                (subjaxpr,
                 map(read, const_vars),
                 map(read, bound_vars))
                for subjaxpr, const_vars, bound_vars in eqn.bound_subjaxprs])
            subfuns = map(lu.wrap_init, subfuns)
            ans, net_params = get_primitive_init(eqn.primitive)(
                prim_rng, eqn.params, sub_consts, sub_freevar_val, in_vals,
                net_params)
        else:
            ans, net_params = get_primitive_init(eqn.primitive)(
                prim_rng, net_params, *in_vals, **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, outvals)
    return net_params

init_rules = {}

def layer_init(layer, rng, net_params, *inputs):
    if layer.name not in net_params:
        layer_params = layer.init_fun(rng, *inputs)
        net_params[layer.name] = layer_params
    return layer.apply_fun(net_params[layer.name], *inputs), net_params

def get_primitive_init(primitive):
    if primitive in init_rules:
        return primitive
    elif isinstance(primitive, Layer):
        return partial(layer_init, primitive)
    else:
        return (lambda _, net_params, *in_vals, **params:
                (primitive.bind(*in_vals, **params), net_params))

def init_fun(net_fun, rng, *example_inputs, **kwargs):
    net_fun = lu.wrap_init(net_fun)
    def pv_like(x):
        return pe.PartialVal((get_aval(x), jc.unit))
    pvals = map(pv_like, example_inputs)
    jaxpr, _, consts = pe.trace_to_jaxpr(net_fun, pvals, **kwargs)
    return init_interpreter(rng, jaxpr, consts, [], {}, *example_inputs)


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
        if isinstance(primitive, Layer):
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
    return Layer(name, init_fun, apply_fun).bind


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
