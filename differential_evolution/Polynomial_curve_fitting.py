# asynchronous DE
# ref: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
# modified by Choi, T

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

x = np.linspace(0, 10, 500)
y = np.cos(x) + np.random.normal(0, 0.2, 500)


def fmodel(x, w):
    return w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 + w[5] * x**5


def rmse(w):
    y_pred = fmodel(x, w)
    return np.sqrt(sum((y - y_pred) ** 2) / len(y))


def de(F, bounds, sf=0.5, cr=0.9, pop_size=100, gen_max=1500):
    dim_size = len(bounds)
    lower, upper = np.asarray(bounds).T

    pop = np.random.rand(pop_size, dim_size)
    pop_denorm = lower + pop * (upper - lower)
    fitness = np.asarray([F(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)

    for gen in range(gen_max):
        for i in range(pop_size):
            # mutation operator
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(r1 + sf * (r2 - r3), 0, 1)

            # crossover operator
            cross_points = np.random.rand(dim_size) < cr
            cross_points[np.random.randint(dim_size)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial_denorm = lower + trial * (upper - lower)

            # selection operator
            f = F(trial_denorm)
            if f <= fitness[i]:
                pop[i] = trial
                fitness[i] = f
                if f <= fitness[best_idx]:
                    best_idx = i

        pop_denorm = lower + pop * (upper - lower)
        yield pop_denorm, fitness, best_idx


result = list(
    de(rmse, [(-5, 5)] * 6, pop_size=50, gen_max=500))

plt.rcParams["figure.figsize"] = (10, 7.5)
plt.rcParams['axes.grid'] = True

pop_25, _, best_25 = result[24]
pop_50, _, best_50 = result[49]
pop_75, _, best_75 = result[74]
pop_100, _, best_100 = result[99]
pop_125, _, best_125 = result[124]

plt.ylim([-2, 2])
plt.scatter(x, y)
plt.plot(x, np.cos(x), label='cos(x)')
plt.plot(x, fmodel(x, pop_25[best_25]), label='gen=25')
plt.plot(x, fmodel(x, pop_50[best_50]), label='gen=50')
plt.plot(x, fmodel(x, pop_75[best_75]), label='gen=75')
plt.plot(x, fmodel(x, pop_100[best_100]), label='gen=100')
plt.plot(x, fmodel(x, pop_125[best_125]), label='gen=125')
plt.legend()
plt.show()
