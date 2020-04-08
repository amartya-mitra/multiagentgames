from jax import jit, grad, vmap, random, jacrev, jacobian, jacfwd
from functools import partial
import jax
import jax.numpy as jp
import jax.scipy as jsp
from jax.experimental import stax # neural network library
from jax.experimental.stax import GeneralConv, Conv, ConvTranspose, Dense, MaxPool, Relu, Flatten, LogSoftmax, LeakyRelu, Dropout, Tanh, Sigmoid, BatchNorm # neural network layers
from jax.nn import softmax, sigmoid
from jax.nn.initializers import zeros
import matplotlib.pyplot as plt # visualization
import numpy as np
from jax.experimental import optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of collections of numpy arrays
from jax.ops import index, index_add, index_update
import os, time

rng=jax.random.PRNGKey(1234)

def naive(Ls, th, eta):
    grad_L = jacobian(Ls)(th) # n x n x d
    grads = jp.einsum('iij->ij', grad_L)
    step = eta * grads
    assert th.shape == step.shape
    return th - step

def la(Ls, th, eta, alpha):
    grad_L = jacobian(Ls)(th) # n x n x d
    def fn1(th):
        xi = jp.lax.stop_gradient(jp.einsum('ii...->i...', jax.jacrev(Ls)(th)))
        _, prod = jax.jvp(Ls, (th,), (xi,))
        return (prod - jp.einsum('ij,ij->i', xi, xi))

    xi = jp.einsum('iij->ij', grad_L)
    grads = xi - alpha * jp.einsum('ii...->i...', jax.jacrev(fn1)(th))
    step = eta * grads
    assert th.shape == step.shape
    return th - step

def lola(Ls, th, eta, alpha):
    grad_L = jacobian(Ls)(th) # n x n x d
    def fn1(th):
        xi = jp.einsum('ii...->i...', jax.jacrev(Ls)(th))
        _, prod = jax.jvp(Ls, (th,), (xi,))
        return (prod - jp.einsum('ij,ij->i', xi, xi))

    xi = jp.einsum('iij->ij', grad_L)
    grads = xi - alpha * jp.einsum('ii...->i...', jax.jacrev(fn1)(th))
    step = eta * grads
    assert th.shape == step.shape
    return th - step

def symlola(Ls, th, eta, alpha):
    grad_L = jacobian(Ls)(th) # n x n x d
    def fn1(th):
        xi = jp.einsum('ii...->i...', jax.jacrev(Ls)(th))
        _, prod = jax.jvp(Ls, (th,), (xi,))
        return prod

    xi = jp.einsum('iij->ij', grad_L)
    grads = xi - alpha * jp.einsum('ii...->i...', jax.jacrev(fn1)(th))
    step = eta * grads
    assert th.shape == step.shape
    return th - step
