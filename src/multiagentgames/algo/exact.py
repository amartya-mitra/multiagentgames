from jax import jacobian, lax
import jax
import jax.numpy as jp
from jax.ops import index, index_add, index_update
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

    def lola0(Ls, th, hp):
        grad_L = jacobian(Ls)(th)  # n x n x d
        xi = jp.einsum('iij->ij', grad_L)
        full_hessian = jax.hessian(Ls)(th)
        diag_hessian = jp.einsum('iijkl->ijkl', full_hessian)
        for i in range(th.shape[0]):
            diag_hessian = index_update(diag_hessian, index[i,:,i,:], 0)
        third_term = jp.einsum('iij->ij',jp.einsum('ijkl,mij->mkl',diag_hessian,grad_L))
        grads = xi - hp['alpha'] * third_term
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

    def sos(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        xi = jp.einsum('iij->ij',grad_L)
        full_hessian = jax.hessian(Ls)(th)
        off_diag_hessian = full_hessian
        for i in range(th.shape[0]):
            off_diag_hessian = index_update(off_diag_hessian, index[i,i,:,:,:], 0)
        second_term = - hp['alpha'] * jp.einsum('iim->im',jp.einsum('ijklm,jk->ilm', off_diag_hessian, xi))
        xi_0 = xi + second_term # n x d
        diag_hessian = jp.einsum('iijkl->ijkl', full_hessian)
        for i in range(th.shape[0]):
            diag_hessian = index_update(diag_hessian, index[i,:,i,:], 0)
        third_term = - hp['alpha'] * jp.einsum('iij->ij',jp.einsum('ijkl,mij->mkl',diag_hessian,grad_L))
        dot = jp.einsum('ij,ij', third_term, xi_0)
        pass_through = lambda x: x
        p1 = lax.cond(dot >= 0, #Condition
                      1.0, pass_through, #True
                      jp.minimum(1, - hp['a'] * jp.linalg.norm(xi_0)**2 / dot), pass_through) #False
        xi_norm = jp.linalg.norm(xi)
        p2 = lax.cond(xi_norm < hp['b'], #Condition
                      xi_norm**2, pass_through, #True
                      1.0, pass_through) #False
        p = jp.minimum(p1, p2)
        grads = xi_0 + p * third_term
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)

    def sga(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        xi = jp.einsum('iij->ij', grad_L)
        full_hessian = jax.hessian(Ls)(th)
        full_hessian_transpose = jp.einsum('ij...->ji...',full_hessian)
        hess_diff = full_hessian - full_hessian_transpose
        second_term = -hp['lambda'] * jp.einsum('iim->im',jp.einsum('ijklm,jk->ilm', hess_diff, xi))
        grads = xi + second_term
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)

