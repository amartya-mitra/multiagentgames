from src.multiagentgames.algo.exact import *
from src.multiagentgames.game.simple import *


'''Iterated Prisoner's Dilemma - SOS/LOLA vs LA/CO/SGA/EG/CGD/LSS/NAIVE'''
rng=jax.random.PRNGKey(1234)

gamma = 0.96
dims, Ls = ipd(gamma)
num_runs = 50
num_epochs = 200
hp={'eta':1.0, 'alpha':1.0}
std = 1
algo_list = ['NAIVE', 'LOLA', 'LA', 'SYMLOLA', 'SOS', 'SGA', 'CO',  'EG', 'CGD', 'LSS' ][0:4]

def init_th(dims, std, rng):
  th = []
  init=random.normal(rng, shape=(jp.sum(jp.array(dims)),))
  if std > 0:
    init*=std
  for i in range(len(dims)):
    th.append(init[:dims[i]])
    init=init[dims[i]:]
  return jp.array(th)

theta = vmap(partial(init_th, dims, std))(jax.random.split(rng, num_runs))

t1 = time.time()
plt.figure(figsize=(15, 8))
for algo in [s.lower() for s in algo_list]:
    losses_out = np.zeros((num_runs, num_epochs))
    update_fn=jit(vmap(partial(Algorithms[algo], Ls), in_axes=(0, None), out_axes=(0, 0)))
    th = theta
    for k in range(num_epochs):
        th, losses = update_fn(th, hp)
        losses_out[:, k] = (1-gamma)*losses[:, 0]
    mean = np.mean(losses_out, axis=0)
    dev = np.std(losses_out, axis=0)
    plt.plot(np.arange(num_epochs), mean)
    plt.fill_between(np.arange(num_epochs), mean-dev, mean+dev, alpha=0.08)

plt.title('IPD Results')
plt.xlabel('Learning Step')
plt.ylabel('Average Loss')
plt.legend(algo_list, loc='upper left', frameon=True, framealpha=1, ncol=3)
plt.show()
print('Jax time:', time.time()-t1)