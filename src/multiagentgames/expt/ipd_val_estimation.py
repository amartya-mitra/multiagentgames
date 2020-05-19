from multiagentgames.game.simple import *
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jp
from jax import jit, grad, vmap, random, jacrev, jacobian, jacfwd, value_and_grad
from functools import partial
from jax.experimental import stax # neural network library
from jax.experimental.stax import GeneralConv, Conv, ConvTranspose, Dense, MaxPool, Relu, Flatten, LogSoftmax, LeakyRelu, Dropout, Tanh, Sigmoid, BatchNorm, Softmax # neural network layers
from jax.nn import softmax, sigmoid
from jax.experimental import optimizers
from collections import deque
from multiagentgames.lib import util

@util.functiontable
class Algos:
    def naive(Ls, th, hp):
        # print(th.shape)
        grad_L = jacobian(Ls)(th)  # n x n x d
        # print(Ls(th).shape, grad_L.shape)

        # grad = Trace(\grad_{\Theta}V(\Theta))
        # Shape: (n,d)
        grads = jp.einsum('xiij->ij', grad_L)
        # print(grads.shape)

        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th).reshape(-1)

    def lola(Ls, th, hp):
        # print(th.shape)
        grad_L = jacobian(Ls)(th)  # n x n x d
        # print(Ls(th).shape, grad_L.shape)

        def fn1(th):
            """ This function returns the second term of the taylor expansion of
                Vi(\theta_1, \theta_2 + \del\theta_2, ..., \theta_n + \del \theta_n).
                    where \del \theta_i = \grad_{\theta_i} Vi(\Theta)

                The terms is [\sum_{j \ne i} \del\theta_j * \grad_{\theta_j} Vi(\Theta)]
                Shape: (n,)
            """
            # xi = Trace(\grad_{\Theta}V(\Theta)) i.e. [\grad_{\theta_i}Vi(\Theta)]
            # Shape: (n,d)
            xi = jp.einsum('xii...->i...', jax.jacrev(Ls)(th))

            # prod = [\sum_i \grad_{\theta_i}Vi(\Theta) \grad_{\theta_i}Vj(\Theta)]
            # Shape: (n,)
            _, prod = jax.jvp(Ls, (th,), (xi,))
            prod = jp.einsum('x...->...', prod)

            # This sets \grad_{\theta_i}Vi(\Theta) \grad_{\theta_i}Vi(\Theta) = 0
            # So, you get [\sum_{j \ne i} \grad_{\theta_j}Vj(\Theta) * \grad_{\theta_j}Vi(\Theta)]
            # Shape: (n,)
            return (prod - jp.einsum('ij,ij->i', xi, xi))

        # xi = [\grad_{\theta1}V1(\Theta),...,\grad_{\theta_n}Vn(\Theta)]
        # Shape: (n,d)
        xi = jp.einsum('xiij->ij', grad_L)

        # The second term here returns
        # [\grad_{\theta_i} (\sum_{j \ne i} \grad_{\theta_j}Vj(\Theta) * \grad_{\theta_j}Vi(\Theta)) ]
        # This is the LOLA term for SOS paper which also considers accounting for action opponent
        # took based on our value function. This is him taking us into the account.
        grads = xi - hp['alpha'] * jp.einsum('ii...->i...', jax.jacrev(fn1)(th))
        # print(grads.shape)
        step = hp['eta'] * grads
        return th - step.reshape(th.shape), Ls(th).reshape(-1)


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def add_batch(self, theta, losses):
        for t,l in zip(theta, losses):
            self.add((t,l))
            print('Size:', self.size())

    def sample(self, batch_size=64):
        idx = np.random.choice(np.arange(len(self.buffer)),
                                size=batch_size,
                                replace=False)
        samplelist =  [self.buffer[ii] for ii in idx]
        thetalist = [k for k,v in samplelist]
        losslist = [v for k,v in samplelist]
        return jp.stack(thetalist), jp.stack(losslist)

class LossNetwork:
    def __init__(self, rng, learning_rate=0.001, nplayers=2,
                 nparams=5, hidden_size=1000,
                 name='Network'):
        self.key = rng
        self.init_fun, self.apply_fun = stax.serial(
            Flatten,
            Dense(hidden_size), Relu,
            Dense(hidden_size), Relu,
            Dense(hidden_size), Relu,
            Dense(nplayers)
        )
        self.in_shape = (-1, nplayers, nparams)
        _, self.net_params = self.init_fun(self.key, self.in_shape)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=learning_rate)
        self.opt_state = self.opt_init(self.net_params)
        self.loss = np.inf

    def loss_fun(self, params, inputs, targets):
        inputs_flatten = inputs.reshape(self.in_shape)
        output = self.apply_fun(params, inputs_flatten)
        return jp.mean(jp.square(output-targets))

    def output(self, inputs):
        inputs_flatten = inputs.reshape(self.in_shape)
        return self.apply_fun(self.net_params, inputs_flatten)

    def step(self, i, inputs, targets):
        params = self.get_params(self.opt_state)
        self.loss, g = value_and_grad(self.loss_fun)(params, inputs, targets)
        self.opt_state = self.opt_update(i, g, self.opt_state)
        self.net_params = self.get_params(self.opt_state)

def init_th(dims, std, rng):
  th = []
  init=random.normal(rng, shape=(jp.sum(jp.array(dims)),))
  if std > 0:
    init*=std
  for i in range(len(dims)):
    th.append(init[:dims[i]])
    init=init[dims[i]:]
  return jp.array(th)


def init_memory(loss_fn, theta, mem_size=1000):
    memory = Memory(max_size=mem_size)
    losses = vmap(loss_fn)(theta)
    memory.add_batch(theta, losses)
    return memory

def train(lossN, memory, max_epochs=500):
    for ep in range(max_epochs):
        batch_theta, batch_losses = memory.sample()
        # print(batch_theta.shape, batch_losses.shape, lossN.output(batch_theta).shape)
        lossN.step(ep, batch_theta, batch_losses)
        loss = lossN.loss_fun(lossN.net_params, batch_theta, batch_losses)
        print(f"Ep:{ep}, Loss:{loss}")

def scatterplot(probs, title):
    colors = cm.rainbow(np.linspace(0, 1, probs.shape[-1]))
    labels = ['s0', 'CC', 'CD', 'DC', 'DD']
    for i, (l, c) in enumerate(zip(labels, colors)):
        plt.scatter(probs[:, 0, i], probs[:, 1, i], color=c, label=l)
    plt.legend()
    plt.title(title)
    plt.xlabel("p(C | state)_agent 1")
    plt.ylabel("p(C | state)_agent 2")
    plt.show()

def main():
    rng = jax.random.PRNGKey(1234)

    gamma = 0.96
    dims, Ls = ipd(gamma)
    num_runs = 100
    num_epochs = 500
    memory_size = 50000
    algo_hp = {
        'naive': {'eta': 1.0},
        'lola': {'eta': 1.0, 'alpha': 1.0},
        'la': {'eta': 1.0, 'alpha': 1.0},
        'sos': {'eta': 1.0, 'alpha': 1.0, 'a': 0.5, 'b': 0.1},
        'co': {'eta': 1.0, 'gamma':50.0},
        'sga': {'eta': 1.0, 'lambda':1.0},
        'cgd': {'eta': 1.0},
        'lolacgd': {'eta': 0.5}
    }
    hp = {}
    std = 1
    # algo_list = ['NAIVE', 'LOLA0', 'LOLA', 'LA', 'SYMLOLA', 'SOS', 'SGA', 'PSGA', 'CO', 'EG', 'CGD', 'LSS'][0:8]
    algo_list = ['NAIVE', 'LOLA', 'LA', 'SOS', 'CO', 'SGA', 'CGD', 'LOLACGD'][1:2]

    mem_theta = vmap(partial(init_th, dims, std))(jax.random.split(rng, memory_size))
    memory = init_memory(Ls, mem_theta, mem_size=memory_size)
    print('Final size:', memory.size())
    # print('Sample:', memory.sample())

    lossN = LossNetwork(rng, learning_rate=1e-2)
    train(lossN, memory, max_epochs=20000)
    # output_val = lossN.output(theta)
    # print(output_val.shape, vmap(Ls)(theta).shape)
    theta = vmap(partial(init_th, dims, std))(jax.random.split(rng, num_runs))
    prob_fn = jit(vmap(sigmoid))

    t1 = time.time()
    plt.figure(figsize=(15, 8))
    probslist = []
    for algo in [s.lower() for s in algo_list]:
        losses_out = np.zeros((num_runs, num_epochs))
        # update_fn=jit(vmap(partial(Algorithms[algo], lossN.output), in_axes=(0, None), out_axes=(0, 0)), static_argnums=1)
        update_fn=vmap(partial(Algos[algo], lossN.output), in_axes=(0, None), out_axes=(0, 0))
        th = theta
        for k,v in algo_hp[algo].items():
            hp[k] = v
        for k in range(num_epochs):
            # th, losses, eig = update_fn(th, hp)
            # eig = jp.abs(eig)
            # print(jp.min(eig), jp.max(eig))
            th, losses = update_fn(th, hp)
            # print(losses.shape)
            losses_out[:, k] = (1-gamma)*losses[:, 0]
        probslist.append(prob_fn(th))
        mean = np.mean(losses_out, axis=0)
        dev = np.std(losses_out, axis=0)
        plt.plot(np.arange(num_epochs), mean)
        plt.fill_between(np.arange(num_epochs), mean-dev, mean+dev, alpha=0.08)

    plt.title('IPD Results')
    plt.xlabel('Learning Step')
    plt.ylabel('Average Loss')
    plt.legend(algo_list, loc='upper left', frameon=True, framealpha=1, ncol=3)
    print('Jax time:', time.time()-t1)
    plt.show()
    for p, a in zip(probslist, algo_list):
        scatterplot(p, a)

if __name__ == "__main__":
    main()