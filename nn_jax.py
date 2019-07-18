"""Jax implementations of functions from nn.py"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from operator import mul

import numpy as onp
import numpy.random as npr

from jax import random
from jax.util import safe_map
from jax.core import Primitive, unit, unitvar, pack, pack_p
from jax.api import make_jaxpr
import jax.experimental.stax as stax
import jax.lax as lax
import jax.numpy as np
from jax.scipy.special import expit as sigmoid, logsumexp


map = safe_map

def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)

def softplus(x):
    return np.logaddexp(0, x)

def concat_elu(x):
    """
    like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU
    """
    return elu(np.concatenate((x, -x), -1))

def log_prob_from_logits(x):
    """numerically stable log_softmax implementation that prevents overflow"""
    x = x - np.max(x, axis=0, keepdims=True)
    return x - logsumexp(x, axis=0, keepdims=True)

def pcnn_out_to_conditional_params(x, theta, nr_mix=10):
    """
    Maps data x and model output theta to conditional parameters for a mixture
    of nr_mix logistics. If the input shapes are

    x.shape == (n, h, w, c)
    theta.shape == (n, h, w, 10 * nr_mix)

    the output shapes will be

    means.shape == log_scales.shape == (nr_mix, n, h, w, c)
    logit_probs.shape == (nr_mix, n, h, w)
    """
    logit_probs, theta = np.split(theta, [nr_mix], axis=-1)
    logit_probs = np.moveaxis(logit_probs, -1, 0)
    theta = np.moveaxis(np.reshape(theta, x.shape + (-1,)), -1, 0)
    unconditioned_means, log_scales, coeffs = np.split(theta, 3)
    coeffs = np.tanh(coeffs)

    # now condition the means for the last 2 channels
    mean_red   = unconditioned_means[..., 0]
    mean_green = unconditioned_means[..., 1] + coeffs[..., 0] * x[..., 0]
    mean_blue = (unconditioned_means[..., 2] + coeffs[..., 1] * x[..., 0]
                 + coeffs[..., 2] * x[..., 1])
    means = np.stack((mean_red, mean_green, mean_blue), axis=-1)
    inv_scales = softplus(log_scales)
    return means, inv_scales, logit_probs

def conditional_params_to_logprob(x, conditional_params):
    means, inv_scales, logit_probs = conditional_params
    cdf = lambda offset: sigmoid((x - means + offset) * inv_scales)
    upper_cdf = np.where(x ==  1, 1, cdf( 1 / 255))
    lower_cdf = np.where(x == -1, 0, cdf(-1 / 255))
    all_logprobs = np.sum(np.log(np.maximum(upper_cdf - lower_cdf, 1e-12)), -1)
    log_mix_coeffs = log_prob_from_logits(logit_probs)
    return np.sum(
        np.mean(logsumexp(log_mix_coeffs + all_logprobs, axis=0), axis=0))


# Initializers
def randn(stddev=1e-2, rng=npr):
    """An initializer function for random normal coefficients."""
    def init(shape):
        return rng.normal(size=shape, scale=stddev).astype('float32')
    return init

zeros = functools.partial(np.zeros, dtype='float32')
ones = functools.partial(np.ones, dtype='float32')


def l2_normalize(x, axis=None, epsilon=1e-12):
    return x / np.sqrt(
        np.maximum(np.sum(x ** 2, axis, keepdims=True), epsilon))

# Layers

# Each layer constructor function returns an (init_fun, apply_fun) pair, where
#   init_fun: takes an example input and returns an (example_output, params) pair,
#   apply_fun: takes params, inputs, and an rng key and applies the layer.
class StaxLayer(tuple):
    def __new__(cls, *args, **kwargs):
        return super(StaxLayer, cls).__new__(cls, tuple(*args, **kwargs))

    def __init__(self, *args, **kwargs):
        self.primitive = Primitive('StaxLayer')
        self.primitive.stax_layer = self
        def impl(*args, **kwargs):
            raise ValueError  # TODO: explain that you're not meant to call this function
        self.primitive.def_impl(impl)
        self.primitive.def_abstract_eval(lambda x: x)

    def __call__(self, inputs):
        if isinstance(inputs, (tuple, list)):
            inputs = pack_p.bind(*inputs)
        return self.primitive.bind(inputs)

pack_p.stax_layer = (
    lambda *example_inputs: (tuple(example_inputs), ()),
    lambda params, *inputs, rng=None: tuple(inputs))


# Some constructors could previously be copied straight from stax
# Dropout = lambda *args, **kwargs: StaxLayer(stax.Dropout(*args, **kwargs))
# serial = lambda *args, **kwargs: StaxLayer(stax.serial(*args, **kwargs))
# parallel = lambda *args, **kwargs: StaxLayer(stax.parallel(*args, **kwargs))
# FanOut = lambda *args, **kwargs: StaxLayer(stax.FanOut(*args, **kwargs))
# Identity = StaxLayer(stax.Identity)

def Dropout(rate, mode='train'):
  """Layer construction function for a dropout layer with given rate."""
  def init_fun(example_inputs):
    return example_inputs, ()
  def apply_fun(params, inputs, **kwargs):
    rng = kwargs.get('rng', None)
    if rng is None:
      msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
             "argument. That is, instead of `apply_fun(params, inputs)`, call "
             "it like `apply_fun(params, inputs, key)` where `key` is a "
             "jax.random.PRNGKey value.")
      raise ValueError(msg)
    if mode == 'train':
      keep = random.bernoulli(rng, rate, inputs.shape)
      return np.where(keep, inputs / rate, 0)
    else:
      return inputs
  return StaxLayer((init_fun, apply_fun))

def serial(*layers):
  """Combinator for composing layers in serial.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
    composition of the given sequence of layers.
  """
  nlayers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fun(example_inputs):
    params = []
    for init_fun in init_funs:
      example_inputs, param = init_fun(example_inputs)
      params.append(param)
    return example_inputs, params
  def apply_fun(params, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
    for fun, param, rng in zip(apply_funs, params, rngs):
      inputs = fun(param, inputs, rng=rng, **kwargs)
    return inputs
  return StaxLayer((init_fun, apply_fun))


def parallel(*layers):
  """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the FanOut and
  FanInSum layers.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the
    parallel composition of the given sequence of layers. In particular, the
    returned layer takes a sequence of inputs and returns a sequence of outputs
    with the same length as the argument `layers`.
  """
  nlayers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fun(example_inputs):
    return zip(*[init(shape) for init, shape in zip(init_funs, example_inputs)])
  def apply_fun(params, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
    return [f(p, x, rng=r, **kwargs) for f, p, x, r in zip(apply_funs, params, inputs, rngs)]
  return StaxLayer((init_fun, apply_fun))

def FanOut(num):
  """Layer construction function for a fan-out layer."""
  init_fun = lambda example_inputs: ([example_inputs] * num, ())
  apply_fun = lambda params, inputs, **kwargs: [inputs] * num
  return StaxLayer((init_fun, apply_fun))

def Identity():
  """Layer construction function for an identity layer."""
  init_fun = lambda example_inputs: (example_inputs, ())
  apply_fun = lambda params, inputs, **kwargs: inputs
  return StaxLayer((init_fun, apply_fun))
Identity = Identity()

def _no_params(fun):
    def init_fun(example_inputs):
        return fun(example_inputs), ()
    def apply_fun(_, inputs, rng=None):
        return fun(inputs)
    return StaxLayer((init_fun, apply_fun))
Elu = _no_params(elu)
ConcatElu = _no_params(concat_elu)
Sigmoid = _no_params(sigmoid)


def Dense(out_dim, init_scale=1.):
    """Layer constructor function for a dense (fully-connected) layer."""
    def init_fun(example_inputs):
        input_shape = np.shape(example_inputs)
        V = randn(0.05)((input_shape[-1], out_dim))
        g = ones((out_dim,))
        b = zeros((out_dim,))

        example_output = apply_fun((V, g, b), example_inputs)

        # m_init, v_init = tf.nn.moments(x, [0])
        m_init = np.mean(example_output, 0, keepdims=False)
        v_init = np.var(example_output, 0, keepdims=False)

        # scale_init = init_scale / tf.sqrt(v_init + 1e-10)
        scale_init = init_scale / np.sqrt(v_init + 1e-10)

        # with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
        #     # x = tf.identity(x)
        #     x = tf.matmul(x_, V)
        #     scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        #     x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])
        g = g * scale_init
        b = b - m_init * scale_init
        return apply_fun((V, g, b), example_inputs), (V, g, b)
    def apply_fun(params, inputs, rng=None):
        V, g, b = params
        # x = tf.matmul(x_, V)
        inputs = np.dot(inputs, V)
        # scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        scaler = g / np.sqrt(np.sum(V ** 2, 0))
        # x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])
        return scaler * inputs + b
    return StaxLayer((init_fun, apply_fun))


def GeneralConv(dimension_numbers, out_chan, filter_shape=[3, 3], strides=None,
                padding='SAME', init_scale=1.):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    # W_init = W_init or glorot(rhs_spec.index('O'), rhs_spec.index('I'))
    def init_fun(example_inputs):
        input_shape = np.shape(example_inputs)
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
        bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
        V = randn(0.05)(kernel_shape)
        g = ones(bias_shape)
        b = zeros(bias_shape)
        example_output = apply_fun((V, g, b), example_inputs)

        # m_init, v_init = tf.nn.moments(x, [0, 1, 2])
        not_chan = [dim for dim, c in enumerate(out_spec) if c != 'C']
        m_init = np.reshape(
            np.mean(example_output, axis=not_chan, keepdims=True), bias_shape)
        v_init = np.reshape(
            np.var(example_output, axis=not_chan, keepdims=True), bias_shape)

        # scale_init = init_scale / tf.sqrt(v_init + 1e-10)
        scale_init = init_scale / np.sqrt(v_init + 1e-10)

        # with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
        g = g * scale_init
        b = b - m_init * scale_init
        return apply_fun((V, g, b), example_inputs), (V, g, b)
    def apply_fun(params, inputs, rng=None):
        V, g, b = params
        not_out_chan = [dim for dim, c in enumerate(rhs_spec) if c != 'O']
        W = g * l2_normalize(V, not_out_chan)
        return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                        dimension_numbers) + b
    return StaxLayer((init_fun, apply_fun))
Conv2D = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))

def DeConv2D(out_chan, filter_shape=[3, 3], strides=None, padding='SAME',
             init_scale=1.):
    """Assumes NHWC format."""
    one = (1,) * len(filter_shape)
    strides = strides or one
    def init_fun(example_inputs):
        input_shape = np.shape(example_inputs)
        filter_shape_iter = iter(filter_shape)
        kernel_shape = tuple(filter_shape) + (out_chan, input_shape[-1])
        bias_shape = [out_chan]
        V = randn(0.05)(kernel_shape)
        g = ones(bias_shape)
        b = zeros(bias_shape)
        example_output = apply_fun((V, g, b), example_inputs)

        # m_init, v_init = tf.nn.moments(x, [0, 1, 2])
        not_chan = [0, 1, 2]
        m_init = np.reshape(
            np.mean(example_output, axis=not_chan, keepdims=True), bias_shape)
        v_init = np.reshape(
            np.var(example_output, axis=not_chan, keepdims=True), bias_shape)

        # scale_init = init_scale / tf.sqrt(v_init + 1e-10)
        scale_init = init_scale / np.sqrt(v_init + 1e-10)

        # with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
        g = g * scale_init
        b = b - m_init * scale_init
        return apply_fun((V, g, b), example_inputs), (V, g, b)
    def apply_fun(params, inputs, rng=None):
        input_shape = np.shape(inputs)
        if padding == 'SAME':
            out_space = onp.multiply(input_shape[1:3], strides)
        elif padding == 'VALID':
            out_space = onp.add(onp.multiply(input_shape[1:3], strides),
                               filter_shape) - 1
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")
        out_shape = (input_shape[0],) + tuple(out_space) + (out_chan,)
        V, g, b = params
        W = g[..., np.newaxis] * l2_normalize(V, [0, 1, 3])
        W_shape = np.shape(W)
        dimension_numbers = lax.conv_dimension_numbers(
            input_shape, W_shape, ('NHWC', 'HWIO', 'NHWC'))
        lhs_perm, rhs_perm, _ = dimension_numbers
        padding_ = lax.padtype_to_pads(out_shape[1:3], W_shape[:2], strides,
                                       padding)
        return lax.lax._conv_general_dilated_transpose_lhs(
            inputs, W, strides, padding_, one, one, dimension_numbers, 1,
            out_shape, W_shape, None) + b
    return StaxLayer((init_fun, apply_fun))

def NIN(num_units, init_scale=1.):
    """1x1 convolution"""
    dense_init, dense_apply = Dense(num_units, init_scale)
    def init_fun(example_inputs):
        *in_shp, in_chan = np.shape(example_inputs)
        example_inputs = np.reshape(example_inputs, (-1, in_chan))
        example_output, params = dense_init(example_inputs)
        return np.reshape(example_output, in_shp + [num_units]), params
    def apply_fun(params, inputs, rng=None):
        *in_shp, in_chan = np.shape(inputs)
        inputs = np.reshape(inputs, (-1, in_chan))
        return np.reshape(dense_apply(params, inputs), in_shp + [num_units])
    return StaxLayer((init_fun, apply_fun))

def GatedResnet(nonlinearity=ConcatElu, conv=Conv2D, dropout_p=0.):
    def make_resnet(example_inputs):
        num_filters = np.shape(example_inputs)[-1]
        main_block = serial(
            nonlinearity,
            conv(num_filters),
            nonlinearity,
            Dropout(1 - dropout_p) if dropout_p > 0 else Identity,
            FanOut(2),
            parallel(conv(num_filters, init_scale=0.1),
                     serial(conv(num_filters, init_scale=0.1), Sigmoid)),
            FanInProd)
        return serial(FanOut(2), parallel(Identity, main_block), FanInSum)
    return input_dependent(make_resnet)

def FanInGatedResnet(nonlinearity=ConcatElu, conv=Conv2D, dropout_p=0.):
    def make_resnet(example_inputs):
        num_filters = np.shape(tuple(example_inputs)[0])[-1]
        main_block = serial(
            parallel(
                serial(
                    nonlinearity,
                    conv(num_filters)),
                serial(
                    nonlinearity,
                    NIN(num_filters))),
            FanInSum,
            nonlinearity,
            Dropout(1 - dropout_p) if dropout_p > 0 else Identity,
            FanOut(2),
            parallel(conv(num_filters, init_scale=0.1),
                     serial(conv(num_filters, init_scale=0.1), Sigmoid)),
            FanInProd)
        return serial(FanOut(2),
                      parallel(
                          serial(Take(0), Identity),
                          main_block),
                      FanInSum)
    return input_dependent(make_resnet)

def input_dependent(make_layer):
    def init_fun(example_inputs):
        return make_layer(example_inputs)[0](example_inputs)
    def apply_fun(params, inputs, rng=None):
        return make_layer(inputs)[1](params, inputs, rng=rng)
    return StaxLayer((init_fun, apply_fun))

def FanInSum():
    """Layer construction function for a fan-in sum layer."""
    init_fun = lambda example_inputs: (sum(example_inputs), ())
    apply_fun = lambda params, inputs, rng=None: sum(inputs)
    return StaxLayer((init_fun, apply_fun))
FanInSum = FanInSum()

def FanInProd():
    """Layer construction function for a fan-in product layer."""
    prod = lambda ls: functools.reduce(mul, ls)
    init_fun = lambda example_inputs: (prod(example_inputs), ())
    apply_fun = lambda params, inputs, rng=None: prod(inputs)
    return StaxLayer((init_fun, apply_fun))
FanInProd = FanInProd()

def FanInConcat(axis=-1):
  """Layer construction function for a fan-in concatenation layer."""
  return _no_params(lambda inputs: np.concatenate(tuple(inputs), axis=axis))

def Take(idx):
    init_fun = lambda example_inputs: (example_inputs[idx], ())
    apply_fun = lambda params, inputs, rng=None: tuple(inputs)[idx]
    return StaxLayer((init_fun, apply_fun))



def Pad(padding_config, padding_value=0.):
    return _no_params(functools.partial(lax.pad, padding_value=padding_value,
                                        padding_config=padding_config))
_zero_pad = (0, 0, 0)
DownShift  = Pad([_zero_pad, (1, -1, 0), _zero_pad,  _zero_pad])
RightShift = Pad([_zero_pad, _zero_pad,  (1, -1, 0), _zero_pad])

def DownShiftedConv2D(
        out_chan, filter_shape=[2, 3], strides=None, **kwargs):
    filter_h, filter_w = filter_shape
    return serial(
        Pad([_zero_pad, (filter_h - 1, 0, 0),
             ((filter_w - 1) // 2, (filter_w - 1) // 2, 0), _zero_pad]),
        Conv2D(out_chan, filter_shape=filter_shape, padding='VALID',
               strides=strides, **kwargs))

def DownShiftedDeConv2D(out_chan, filter_shape=[2, 3], strides=None,
                        **kwargs):
    filter_h, filter_w = filter_shape
    return serial(
        DeConv2D(out_chan, filter_shape=filter_shape, padding='VALID',
                 strides=strides, **kwargs),
        Pad([_zero_pad, (0, -(filter_h - 1), 0),
             (-((filter_w - 1) // 2), -((filter_w - 1) // 2), 0), _zero_pad]))


def DownRightShiftedConv2D(
        out_chan, filter_shape=[2, 2], strides=None, **kwargs):
    return serial(
        Pad([_zero_pad, (filter_shape[0] - 1, 0, 0),
             (filter_shape[1] - 1, 0, 0), _zero_pad]),
        Conv2D(out_chan, filter_shape=filter_shape, strides=strides,
             padding='VALID', **kwargs))

def DownRightShiftedDeConv2D(
        out_chan, filter_shape=[2, 2], strides=None, **kwargs):
    return serial(
        DeConv2D(out_chan, filter_shape=filter_shape, strides=strides,
                 padding='VALID', **kwargs),
        Pad([_zero_pad, (0, -(filter_shape[0] - 1), 0),
             (0, -(filter_shape[1] - 1), 0), _zero_pad]))

def pointy_to_stax_layer(fun):
  jaxpr = make_jaxpr(fun)(0)  # Trace on a dummy value
  def init_fun(*example_inputs):
    def read(v):
      return env[v]

    def write(v, val):
      env[v] = val

    env = {}
    params = {}
    write(unitvar, unit)
    assert not jaxpr.constvars
    assert not jaxpr.freevars
    map(write, jaxpr.invars, example_inputs)
    for eqn in jaxpr.eqns:
      example_inputs = map(read, eqn.invars)
      assert not eqn.bound_subjaxprs
      init_fun, _ = eqn.primitive.stax_layer
      ans, param = init_fun(*example_inputs)
      outvals = list(ans) if eqn.destructure else [ans]
      map(write, eqn.outvars, outvals)
      params[id(eqn)] = param
    return read(jaxpr.outvar), params
  def apply_fun(params, inputs, rng=None):
    def read(v):
      return env[v]

    def write(v, val):
      env[v] = val

    env = {}
    write(unitvar, unit)
    assert not jaxpr.constvars
    assert not jaxpr.freevars
    map(write, jaxpr.invars, [inputs])
    for eqn in jaxpr.eqns:
      inputs = map(read, eqn.invars)
      assert not eqn.bound_subjaxprs
      _, apply_fun = eqn.primitive.stax_layer
      rng, subrng = random.split(rng) if rng is not None else (None, None)
      ans = apply_fun(params[id(eqn)], *inputs, rng=subrng)
      outvals = list(ans) if eqn.destructure else [ans]
      map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar)
  return StaxLayer((init_fun, apply_fun))
