from multiagentgames.algo.exact import *
from multiagentgames.game.simple import *


'''Iterated Prisoner's Dilemma - SOS/LOLA vs LA/CO/SGA/EG/CGD/LSS/NAIVE'''

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

    gamma = 0.999
    dims, Ls = simplified_dixit(gamma)
    num_runs = 1000
    num_epochs = 100
    hp = {'eta': 1.0, 'alpha': 1.0, 'a': 0.5, 'b': 0.5, 'lambda':1.0}
    std = 1
    # algo_list = ['NAIVE', 'LOLA0', 'LOLA', 'LA', 'SYMLOLA', 'SOS', 'SGA', 'PSGA', 'CO', 'EG', 'CGD', 'LSS'][0:8]
    algo_list = ['NAIVE', 'LOLA', 'CGD', 'LA', 'SOS', 'SGA', ]

    def split_rng(rng, num_runs): return (rng if num_runs < 2 else jax.random.split(rng, num_runs))

    theta = vmap(partial(init_th, dims, std))(split_rng(rng, num_runs))
    # theta = partial(init_th, dims, std)(split_rng(rng, num_runs))

    t1 = time.time()
    plt.figure(figsize=(15, 8))
    for algo in [s.lower() for s in algo_list]:
        losses_out = np.zeros((num_runs, num_epochs))
        update_fn=jit(vmap(partial(Algorithms[algo], Ls), in_axes=(0, None), out_axes=(0, 0)))
        # update_fn=vmap(partial(Algorithms[algo], Ls), in_axes=(0, None), out_axes=(0, 0))

        th = theta
        for k in range(num_epochs):
            th, losses = update_fn(th, hp)
            losses_out[:, k] = (1-gamma)*losses[:, 0]
            # losses_out[:, k] = losses[:, 0]

        mean = np.mean(losses_out, axis=0)
        dev = np.std(losses_out, axis=0)
        print(f'{algo}: {mean}')
        plt.plot(np.arange(num_epochs), mean)
        plt.fill_between(np.arange(num_epochs), mean-dev, mean+dev, alpha=0.08)

    plt.title('IPD Results')
    plt.xlabel('Learning Step')
    plt.ylabel('Average Loss')
    plt.legend(algo_list, loc='upper left', frameon=True, framealpha=1, ncol=3)
    print('Jax time:', time.time()-t1)
    plt.show()

if __name__ == "__main__":
    main()