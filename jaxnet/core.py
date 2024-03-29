from functools import reduce
import jax.numpy as np
import jax.linear_util as lu
from jax.util import unzip2, unzip3, safe_zip, safe_map, partial, WrapHashably
from jax.abstract_arrays import ShapedArray
from jax.experimental import stax
from jax.interpreters import partial_eval as pe
from jax.interpreters.batching import get_aval
from jax.interpreters import batching
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
        def layer_batch(batched_args, batch_dims, **params):
            # TODO: figure out batching/debatching for init_fun
            batched_apply_fun = (
                lambda params, *batch_inputs:
                batching.batch(lu.wrap_init(partial(self.apply_fun, params)),
                               batch_inputs, batch_dims, 0))
            batched_layer = Layer(name, init_fun, batched_apply_fun)
            return batched_layer.bind(*batched_args, **params), 0
        batching.primitive_batchers[self] = layer_batch

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
            ans, net_params = get_primitive_init(eqn.primitive)(
                prim_rng, eqn.params, sub_consts, sub_freevar_vals, in_vals,
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
        vals_in, net_params = unzip2((t.val, t.net_params) for t in tracers)
        net_params = merge_params(net_params)
        if isinstance(primitive, Layer):
            apply_fun = primitive.apply_fun
            layer_params = net_params[primitive.name]
            return ApplyTracer(
                self, net_params, apply_fun(layer_params, *vals_in))
        else:
            return ApplyTracer(
                self, net_params, primitive.bind(*vals_in, **params))

    def process_call(self, call_primitive, f, tracers, params):
        if call_primitive in pe.map_primitives:
            raise NotImplementedError
        vals, net_params = unzip2((t.val, t.net_params) for t in tracers)
        if any(net_params):
            net_params = merge_params(net_params)
            f = apply_subtrace(f, self.master, WrapHashably(net_params))
            val_out = call_primitive.bind(f, *vals, **params)
            return ApplyTracer(self, net_params, val_out)
        else:
            return call_primitive.bind(f, *vals, **params)


@lu.transformation
def apply_transform(net_params, inputs):
    with jc.new_master(ApplyTrace) as master:
        trace = ApplyTrace(master, jc.cur_sublevel())
        ans = yield map(partial(ApplyTracer, trace, net_params), inputs), {}
        out_tracer = trace.full_raise(ans)
        out_val = out_tracer.val
        del master, out_tracer
    yield out_val

@lu.transformation
def apply_subtrace(master, net_params, *vals):
    net_params = net_params.val
    trace = ApplyTrace(master, jc.cur_sublevel())
    ans = yield map(partial(ApplyTracer, trace, net_params), vals), {}
    out_tracer = trace.full_raise(ans)
    yield out_tracer.val


def apply_fun(net_fun, params, *inputs):
    return apply_transform(lu.wrap_init(net_fun), params).call_wrapped(inputs)
