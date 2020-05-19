from jax import jacobian, lax
import jax
import jax.numpy as jp
from jax.ops import index, index_add, index_update
from multiagentgames.lib import util

rng=jax.random.PRNGKey(1234)

# Let n be the number of players, and
# d be the number of parameters of each player.
# Let \Theta = [\theta_1,.., \theta_n] Shape: (n, d)
# Let V(\Theta) = [V1(\Theta), ..., Vn(\Theta)] Shape: (n,)
# Ls = V(\Theta) Shape: (n,)
# th = \Theta Shape: (n,d) 
# hp = a dict of hyperparams needed to run the experiments.

# grad_L = \grad_{\Theta}V(\Theta)
# Shape: (n, n, d)

@util.functiontable
class Algorithms:
    def naive(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        
        # grad = Trace(\grad_{\Theta}V(\Theta))
        # Shape: (n,d)
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

        # xi = Trace(\grad_{\Theta}V(\Theta)) i.e., \grad_{\theta_i}(Vi(\Theta)
        # Shape: (n,d)
        xi = jp.einsum('iij->ij', grad_L)
        
        # full_hessian = \grad_{\Theta}(\grad_{\Theta}(V(\Theta))
        # Shape: (n, n, d, n, d)
        full_hessian = jax.hessian(Ls)(th)
        
        # diag_hessian = Trace(\grad_{\Theta}(\grad_{\Theta}(V(\Theta)))
        # Shape: (n, d, n, d)
        # Trace was along the V dimension, so this is [\grad_{\theta_j}(\grad_{\theta_i}(Vi(\Theta))]
        # [[\grad_{\theta_1}\grad_{\theta_1}V1(\Theta),...,\grad_{\theta_1}\grad_{\theta_n}Vn(\Theta)], 
        #  [\grad_{\theta_2}\grad_{\theta_1}V1(\Theta),...,\grad_{\theta_2}\grad_{\theta_n}Vn(\Theta)], 
        #                                             ,...,                                          ],
        #  [\grad_{\theta_n}\grad_{\theta_1}V1(\Theta),...,\grad_{\theta_n}\grad_{\theta_n}Vn(\Theta)]]
        diag_hessian = jp.einsum('iijkl->ijkl', full_hessian)

        for i in range(th.shape[0]):
            # Set all \grad_{\theta_i}\grad_{\theta_i}Vi(\Theta) = 0.
            diag_hessian = index_update(diag_hessian, index[i,:,i,:], 0)

        # This term is [\sum_{j \ne i} 
        #                   \grad_{\theta_j} Vi(\Theta) * \grad_{\theta_i}(\grad_{\theta_j}(Vj(\Theta))]
        # Shape: (n,d)
        third_term = jp.einsum('iij->ij',jp.einsum('ijkl,mij->mkl',diag_hessian,grad_L))

        grads = xi - hp['alpha'] * third_term
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)

    def lola(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        def fn1(th):
            """ This function returns the second term of the taylor expansion of 
                Vi(\theta_1, \theta_2 + \del\theta_2, ..., \theta_n + \del \theta_n). 
                    where \del \theta_i = \grad_{\theta_i} Vi(\Theta)

                The terms is [\sum_{j \ne i} \del\theta_j * \grad_{\theta_j} Vi(\Theta)]
                Shape: (n,)
            """
            # xi = Trace(\grad_{\Theta}V(\Theta)) i.e. [\grad_{\theta_i}Vi(\Theta)]
            # Shape: (n,d)
            xi = jp.einsum('ii...->i...', jax.jacrev(Ls)(th))

            # prod = [\sum_i \grad_{\theta_i}Vi(\Theta) \grad_{\theta_i}Vj(\Theta)]
            # Shape: (n,)
            _, prod = jax.jvp(Ls, (th,), (xi,))

            # This sets \grad_{\theta_i}Vi(\Theta) \grad_{\theta_i}Vi(\Theta) = 0
            # So, you get [\sum_{j \ne i} \grad_{\theta_j}Vj(\Theta) * \grad_{\theta_j}Vi(\Theta)]
            # Shape: (n,)
            return (prod - jp.einsum('ij,ij->i', xi, xi))
        
        # xi = [\grad_{\theta1}V1(\Theta),...,\grad_{\theta_n}Vn(\Theta)]
        # Shape: (n,d)
        xi = jp.einsum('iij->ij', grad_L)

        # The second term here returns 
        # [\grad_{\theta_i} (\sum_{j \ne i} \grad_{\theta_j}Vj(\Theta) * \grad_{\theta_j}Vi(\Theta)) ]
        # This is the LOLA term for SOS paper which also considers accounting for action opponent
        # took based on our value function. This is him taking us into the account.
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

    def co(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        xi = jp.einsum('iij->ij', grad_L)
        full_hessian = jax.hessian(Ls)(th)
        full_hessian_transpose = jp.einsum('ij...->ji...',full_hessian)
        second_term = hp['gamma'] * jp.einsum('iim->im',jp.einsum('ijklm,jk->ilm', full_hessian_transpose, xi))
        grads = xi + second_term
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

    ''' Amartya's interpolated SGA '''
    def psga(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        xi = jp.einsum('iij->ij', grad_L)
        full_hessian = jax.hessian(Ls)(th)
        full_hessian_transpose = jp.einsum('ij...->ji...',full_hessian)
        hess_diff = full_hessian - full_hessian_transpose
        second_term = -hp['lambda'] * jp.einsum('iim->im',jp.einsum('ijklm,jk->ilm', hess_diff, xi))
        xi_0 = xi + second_term
        rho = jp.stack(th.shape[0] * [xi], axis=1) + grad_L
        diag_hessian = jp.einsum('iijkl->ijkl', full_hessian)
        for i in range(th.shape[0]):
            diag_hessian = index_update(diag_hessian, index[i,:,i,:], 0)
        third_term = - hp['lambda'] * jp.einsum('iij->ij', jp.einsum('ijkl,mij->mkl', diag_hessian, rho))
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

    def cgd(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        # xi =  Trace(\grad_{\Theta}(V(\Theta))
        # xi = [\grad_{\theta1}V1(\Theta),...,\grad_{\theta_n}Vn(\Theta)]
        # Shape: (n,d)
        xi = jp.einsum('iij->ij', grad_L)

        # full_hessian = \grad_{\Theta}(\grad_{\Theta}(V(\Theta))
        # Shape: (n, n, d, n, d)
        full_hessian = jax.hessian(Ls)(th)

        # diag_hessian = Trace(\grad_{\Theta}(\grad_{\Theta}(V(\Theta)))
        # Shape: (n, d, n, d)
        # Trace was along the V dimension, so this is [\grad_{\theta_j}(\grad_{\theta_i}(Vi(\Theta))]
        # [[\grad_{\theta_1}\grad_{\theta_1}V1(\Theta),...,\grad_{\theta_1}\grad_{\theta_n}Vn(\Theta)], 
        #  [\grad_{\theta_2}\grad_{\theta_1}V1(\Theta),...,\grad_{\theta_2}\grad_{\theta_n}Vn(\Theta)], 
        #                                             ,...,                                          ],
        #  [\grad_{\theta_n}\grad_{\theta_1}V1(\Theta),...,\grad_{\theta_n}\grad_{\theta_n}Vn(\Theta)]]
        diag_hessian = jp.einsum('iijkl->ijkl', full_hessian)


        for i in range(th.shape[0]):
            # Set all \grad_{\theta_i}\grad_{\theta_i}Vi(\Theta) = 0.
            diag_hessian = index_update(diag_hessian, index[i,:,i,:], 0)

        diag_hessian = hp['eta'] *  diag_hessian

        # This term is the competitve term.
        # [\sum_{j \ne i} \grad_{\theta_i}(\grad_{\theta_j}(Vi(\Theta)) * \grad_{\theta_j} Vj(\Theta) ]
        # Shape: (n,d)
        competitive_term = jp.einsum('ikjl, jl-> ik', diag_hessian, xi)

        lola_adjoint_term = xi - competitive_term

        n, dim = th.shape
        Id = jp.stack([jp.eye(dim)]*n, axis=0)
        # This is the equilibrium term from CGD paper. 
        # (Id_{dXd) + \eta^{2} * \grad_{\theta_i}(\grad_{\theta_j}(Vi(\Theta)) * \grad_{\theta_j}(\grad_{\theta_i}(Vj(\Theta)))^{-1}
        equilibrium_term = jp.linalg.inv(Id - jp.einsum('ikjl,jlin->ikn', diag_hessian, diag_hessian))

        grads = jp.einsum('ikl,il -> ik', equilibrium_term, lola_adjoint_term)
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)

    def lolacgd(Ls, th, hp):
        grad_L = jacobian(Ls)(th) # n x n x d
        xi = jp.einsum('iij->ij', grad_L)
        full_hessian = hp['eta'] * jax.hessian(Ls)(th)

        off_diag_hessian = full_hessian
        for i in range(th.shape[0]):
            off_diag_hessian = index_update(off_diag_hessian, index[i,i,:,:,:], 0)
        la_term = jp.einsum('iim->im',jp.einsum('ijklm,jk->ilm', off_diag_hessian, xi))

        diag_hessian = jp.einsum('iijkl->ijkl', full_hessian)
        for i in range(th.shape[0]):
            diag_hessian = index_update(diag_hessian, index[i,:,i,:], 0)
        lola_term = jp.einsum('iij->ij',jp.einsum('ijkl,mij->mkl',diag_hessian,grad_L))

        numerator = xi - la_term - lola_term
        n, dim = th.shape
        Id = jp.stack([jp.eye(dim)]*n, axis=0)
        inverse = jp.linalg.inv(Id - 3 * jp.einsum('ikjl,jlin->ikn', diag_hessian, diag_hessian))
        grads = jp.einsum('ikl,il -> ik', inverse, numerator)
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th)#, jp.linalg.eig(Id - 3 * jp.einsum('ikjl,jlin->ikn', diag_hessian, diag_hessian))[0]
