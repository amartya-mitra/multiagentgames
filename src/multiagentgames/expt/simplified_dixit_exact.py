from multiagentgames.algo.exact import *
from multiagentgames.game.simple import *
import matplotlib.cm as cm

'''Iterated Simplified Dixit - SOS/LOLA vs LA/CO/SGA/EG/CGD/LSS/NAIVE'''

def init_th(dims, std, rng):
  th = []
  init=random.normal(rng, shape=(jp.sum(jp.array(dims)),))
  if std > 0:
    init*=std
  for i in range(len(dims)):
    th.append(init[:dims[i]])
    init=init[dims[i]:]
  return jp.array(th)

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

    gamma = 0.999
    dims, Ls = simplified_dixit(gamma)
    num_runs = 100
    num_epochs = 200
    algo_hp = {
        'naive': {'eta': 100.0},
        'lola': {'eta': 100, 'alpha': 0.0001},
        'la': {'eta': 100, 'alpha':  0.00001},
        'sos': {'eta': 100, 'alpha': 0.0005, 'a': 0.2, 'b': 0.8},
        'co': {'eta': 100, 'gamma':0.01},
        'sga': {'eta': 100, 'lambda':0.0001},
        'cgd': {'eta': 1.0}
    }
    hp = {}
    std = 1
    # algo_list = ['NAIVE', 'LOLA0', 'LOLA', 'LA', 'SYMLOLA', 'SOS', 'SGA', 'PSGA', 'CO', 'EG', 'CGD', 'LSS'][0:8]
    algo_list = [ 'NAIVE', 'LOLA', 'LA', 'SOS', 'CO', 'SGA', 'CGD',]

    def split_rng(rng, num_runs): return (rng if num_runs < 2 else jax.random.split(rng, num_runs))

    theta = vmap(partial(init_th, dims, std))(split_rng(rng, num_runs))
    # theta = partial(init_th, dims, std)(split_rng(rng, num_runs))
    prob_fn = jit(vmap(sigmoid))

    t1 = time.time()
    plt.figure(figsize=(15, 8))
    probslist = []
    for algo in [s.lower() for s in algo_list]:
        losses_out = np.zeros((num_runs, num_epochs))
        update_fn=jit(vmap(partial(Algorithms[algo], Ls), in_axes=(0, None), out_axes=(0, 0)))
        # update_fn=vmap(partial(Algorithms[algo], Ls), in_axes=(0, None), out_axes=(0, 0))

        th = theta
        for k,v in algo_hp[algo].items():
            hp[k] = v
        for k in range(num_epochs):
            th, losses = update_fn(th, hp)
            losses_out[:, k] = (1-gamma)*losses[:, 0]
            # losses_out[:, k] = losses[:, 0]
        probslist.append(prob_fn(th))

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
    for p, a in zip(probslist, algo_list):
        scatterplot(p, a)


if __name__ == "__main__":
    main()