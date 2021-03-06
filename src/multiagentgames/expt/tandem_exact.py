from multiagentgames.algo.exact import *
from multiagentgames.game.simple import *
import random as rnd
import itertools

'''Tandem Game - SOS vs LOLA'''

def init_th(dims, std, rng):
  th = []
  init=random.normal(rng, shape=(jp.sum(jp.array(dims)),))
  if std > 0:
    init*=std
  for i in range(len(dims)):
    th.append(init[:dims[i]])
    init=init[dims[i]:]
  return jp.array(th)

def main():
    rng = jax.random.PRNGKey(1234)
    dims, Ls = tandem()

    num_runs = 100
    num_epochs = 30
    algo_hp = {
        'naive': {'eta': 0.1},
        'lola': {'eta': 0.1, 'alpha': 0.1},
        'la': {'eta': 0.1, 'alpha': 0.1},
        'sos': {'eta': 0.1, 'alpha': 0.1, 'a': 0.5, 'b': 0.5},
        'co': {'eta': 0.0005, 'gamma':100.0},
        'sga': {'eta': 0.1, 'lambda':100.0},
        'cgd': {'eta': 1.0},
        'lolacgd': {'eta': 0.5}
    }
    hp = {}
    std = 0.1
    # algo_list = ['NAIVE', 'LOLA0', 'LOLA', 'LA', 'SYMLOLA', 'SOS', 'SGA', 'PSGA', 'CO', 'EG', 'CGD', 'LSS'][0:9]
    algo_list = ['NAIVE', 'LOLA', 'LA', 'SOS', 'CO', 'SGA', 'CGD', 'LOLACGD']

    theta = vmap(partial(init_th, dims, std))(jax.random.split(rng, num_runs))

    t1 = time.time()
    for algo in [s.lower() for s in algo_list]:
        losses_out = np.zeros((num_runs, num_epochs))
        update_fn = jit(vmap(partial(Algorithms[algo], Ls), in_axes=(0, None), out_axes=(0, 0)), static_argnums=1)
        th = theta
        for k,v in algo_hp[algo].items():
            hp[k] = v
        for k in range(num_epochs):
            th, losses = update_fn(th, hp)
            losses_out[:, k] = losses[:,0]

        mean = np.mean(losses_out, axis=0)
        dev = np.std(losses_out, axis=0)
        plt.plot(np.arange(num_epochs), mean)
        plt.fill_between(np.arange(num_epochs), mean-dev, mean+dev, alpha=0.1)

    plt.title('Tandem Game Results')
    plt.xlabel('Learning Step')
    plt.ylabel('Average Loss')
    plt.legend(algo_list, loc='upper left', frameon=True, framealpha=1)
    print('Jax time:', time.time()-t1)
    plt.show()

if __name__ == "__main__":
    main()