import numpy as onp
from jax import random
from model_jax import PixelCNNPP


# Make a small PixelCNN++
init_fun, apply_fun = PixelCNNPP(nr_resnet=0, nr_filters=2)

#                                n, h, w, c
example_input = onp.random.randn(1, 4, 4, 3)

# Weightnorm (data dependent) initialization
_, params = init_fun(example_input)

example_output = apply_fun(params, example_input, random.PRNGKey(0))
