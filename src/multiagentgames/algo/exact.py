from jax import jacobian
import jax
import jax.numpy as jp
from multiagentgames.lib import util

rng=jax.random.PRNGKey(1234)

@util.functiontable
class Algorithms:
    def naive(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        grads = jp.einsum('iij->ij', grad_L)
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)

    def la(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        def fn1(th):
            xi = jp.lax.stop_gradient(jp.einsum('ii...->i...', jax.jacrev(Ls)(th)))
            _, prod = jax.jvp(Ls, (th,), (xi,))
            return (prod - jp.einsum('ij,ij->i', xi, xi))

        xi = jp.einsum('iij->ij', grad_L)
        grads = xi - hp['alpha'] * jp.einsum('ii...->i...', jax.jacrev(fn1)(th))
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)

    def lola(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        def fn1(th):
            xi = jp.einsum('ii...->i...', jax.jacrev(Ls)(th))
            _, prod = jax.jvp(Ls, (th,), (xi,))
            return (prod - jp.einsum('ij,ij->i', xi, xi))

        xi = jp.einsum('iij->ij', grad_L)
        grads = xi - hp['alpha'] * jp.einsum('ii...->i...', jax.jacrev(fn1)(th))
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)

    def symlola(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        def fn1(th):
            xi = jp.einsum('ii...->i...', jax.jacrev(Ls)(th))
            _, prod = jax.jvp(Ls, (th,), (xi,))
            return prod

        xi = jp.einsum('iij->ij', grad_L)
        grads = xi - hp['alpha'] * jp.einsum('ii...->i...', jax.jacrev(fn1)(th))
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)
