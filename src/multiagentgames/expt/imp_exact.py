from multiagentgames.algo.exact import *
from multiagentgames.game.simple import *

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

def main():
    rng = jax.random.PRNGKey(1234)
    dims, Ls = imp()

    num_runs = 100
    num_epochs = 500
    hp = {'eta': 0.1, 'alpha': 10.0, 'a': 0.5, 'b': 0.5, 'lambda':1.0}
    std = 1
#    algo_list = ['NAIVE', 'LOLA0', 'LOLA', 'LA', 'SYMLOLA', 'SOS', 'SGA', 'PSGA', 'CO', 'EG', 'CGD', 'LSS'][0:8]
    algo_list = ['NAIVE', 'LOLA', 'LA', 'SOS', 'SGA', 'CGD']

    theta = vmap(partial(init_th, dims, std))(jax.random.split(rng, num_runs))

    t1 = time.time()
    plt.figure(figsize=(15, 8))
    for algo in [s.lower() for s in algo_list]:
        losses_out = np.zeros((num_runs, num_epochs))
        update_fn = jit(vmap(partial(Algorithms[algo], Ls), in_axes=(0, None), out_axes=(0, 0)))
        th = theta
        for k in range(num_epochs):
            th, losses = update_fn(th, hp)
            losses_out[:, k] = losses[:,0]

        new_loss = [np.linalg.norm(loss) for loss in jp.mean(losses_out, axis=0)]
        plt.plot(new_loss)

    plt.yscale('log')
    plt.title('Iterated Matching Pennies Results')
    plt.xlabel('Learning Step')
    plt.ylabel('Loss (log scale)')
    plt.legend(algo_list, loc='best', frameon=True, framealpha=1, ncol=3)
    print('Jax time:', time.time() - t1)
    plt.show()

if __name__ == "__main__":
    main()