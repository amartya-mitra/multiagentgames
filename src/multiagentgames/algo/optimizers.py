import jax
import jax.numpy as jp
from jax.util import partial, safe_zip, safe_map, unzip2
from jax import tree_util
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)
from jax import jacobian, lax
from jax.ops import index, index_add, index_update
from multiagentgames.lib import util
from jax import jit, grad, vmap, random, jacrev, jacobian, jacfwd
import functools
from functools import partial

@util.functiontable
class Optimizer:

    def rmsprop(hp):
        def init_fn(th):
            # th is parameter pairs (gen_params, disc_params)
            theta1, theta2 = th
            theta1_flat, theta1_tree = tree_flatten(theta1)
            theta2_flat, theta2_tree = tree_flatten(theta2)
            avg_sq_grad1 = [jp.zeros_like(x) for x in theta1_flat]
            avg_sq_grad2 = [jp.zeros_like(x) for x in theta2_flat]
            return th, (avg_sq_grad1, avg_sq_grad2)

        def update_fn(i, Ls, state):
            th, avg_sq_grad = state

            # th is parameter pairs (gen_params, disc_params)
            theta1, theta2 = th
            theta1_flat, theta1_tree = tree_flatten(theta1)
            theta2_flat, theta2_tree = tree_flatten(theta2)

            # Ls is pair of loss functions (g_loss, d_loss)
            l1, l2 = Ls

            # compute gradients
            grad1, grad2 = grad(l1)(theta1), grad(l2)(theta2)
            grad1_flat, grad1_tree = tree_flatten(grad1)
            grad2_flat, grad2_tree = tree_flatten(grad2)

            # update avg sq grad
            avg_sq_grad1 = [a * hp['gamma'] + g**2 * (1. - hp['gamma'])
                            for g, a in zip(grad1_flat, avg_sq_grad[0])]
            avg_sq_grad2 = [a * hp['gamma'] + g**2 * (1. - hp['gamma'])
                            for g, a in zip(grad2_flat, avg_sq_grad[1])]

            # update parameters
            theta1_flat = [x - hp['step_size'] * g / jp.sqrt(a + hp['eps'])
                           for x, g, a in zip(theta1_flat, grad1_flat, avg_sq_grad1)]
            theta2_flat = [x - hp['step_size'] * g / jp.sqrt(a + hp['eps'])
                           for x, g, a in zip(theta2_flat, grad2_flat, avg_sq_grad2)]

            theta1 = tree_unflatten(theta1_tree, theta1_flat)
            theta2 = tree_unflatten(theta2_tree, theta2_flat)

            return (theta1, theta2), (avg_sq_grad1, avg_sq_grad2)

        def get_params_fn(state):
            return state[0]

        return init_fn, update_fn, get_params_fn

    def naive(hp):
        def init_fn(th):
            return th

        def update_fn(i, Ls, state):
            th = state

            # th is parameter pairs (gen_params, disc_params)
            theta1, theta2 = th
            theta1_flat, theta1_tree = tree_flatten(theta1)
            theta2_flat, theta2_tree = tree_flatten(theta2)

            # Ls is pair of loss functions (g_loss, d_loss)
            l1, l2 = Ls

            # compute gradients
            grad1, grad2 = jacobian(l1)(theta1), jacobian(l2)(theta2)
            grad1_flat, grad1_tree = tree_flatten(grad1)
            grad2_flat, grad2_tree = tree_flatten(grad2)
            step1 = [hp['eta'] * g for g in grad1_flat]
            step2 = [hp['eta'] * g for g in grad2_flat]

            # update parameters
            theta1_flat = [t - s for t, s in zip(theta1_flat, step1)]
            theta2_flat = [t - s for t, s in zip(theta2_flat, step2)]

            theta1 = tree_unflatten(theta1_tree, theta1_flat)
            theta2 = tree_unflatten(theta2_tree, theta2_flat)

            return (theta1, theta2)

        def get_params_fn(state):
            return state

        return init_fn, update_fn, get_params_fn

