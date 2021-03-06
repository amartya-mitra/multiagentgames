from multiagentgames.algo.exact import *
from multiagentgames.game.simple import *
import itertools
import random as rnd

'''Iterated Matching Pennies Game - Convergence Analysis'''

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
    plt.scatter(probs[:, 0], probs[:, 1], color='red')
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.xlabel("p(head)_agent 1")
    plt.ylabel("p(head)_agent 2")
    plt.show()


def main():
    rng = jax.random.PRNGKey(1234)
    dims, Ls = imp()

    num_runs = 100
    num_epochs = 500
    algo_hp = {
        'naive': {'eta': 0.1},
        'lola': {'eta': 0.05, 'alpha': 50.0},
        'la': {'eta': 0.05, 'alpha': 100.0},
        'sos': {'eta': 1.0, 'alpha': 10.0, 'a': 0.5, 'b': 0.7},
        'co': {'eta': 50.0, 'gamma':0.5},
        'sga': {'eta': 0.5, 'lambda':10.0},
        'cgd': {'eta': 0.1}
    }
    std = 1
#    algo_list = ['NAIVE', 'LOLA0', 'LOLA', 'LA', 'SYMLOLA', 'SOS', 'SGA', 'PSGA', 'CO', 'EG', 'CGD', 'LSS'][0:8]
    algo_list = ['NAIVE', 'LOLA', 'LA', 'SOS', 'CO', 'SGA', 'CGD']

    theta = vmap(partial(init_th, dims, std))(jax.random.split(rng, num_runs))
    prob_fn = jit(vmap(sigmoid))

    t1 = time.time()
    plt.figure(figsize=(15, 8))
    probslist = []
    for algo in [s.lower() for s in algo_list]:
        losses_out = np.zeros((num_runs, num_epochs))
        update_fn = jit(vmap(partial(Algorithms[algo], Ls), in_axes=(0, None), out_axes=(0, 0)), static_argnums=1)
        th = theta
        hp = algo_hp[algo]
        for k in range(num_epochs):
            th, losses = update_fn(th, hp)
            losses_out[:, k] = losses[:,0]

        probslist.append(prob_fn(th))
        new_loss = [np.linalg.norm(loss) for loss in jp.mean(losses_out, axis=0)]
        plt.plot(new_loss)

    plt.yscale('log')
    plt.title('Iterated Matching Pennies Results')
    plt.xlabel('Learning Step')
    plt.ylabel('Loss (log scale)')
    plt.legend(algo_list, loc='best', frameon=True, framealpha=1, ncol=3)
    print('Jax time:', time.time() - t1)
    plt.show()
    for p, a in zip(probslist, algo_list):
        scatterplot(p, a)

if __name__ == "__main__":
    main()