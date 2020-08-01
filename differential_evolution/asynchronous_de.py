# asynchronous DE
# ref: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
# modified by Choi, T

import numpy as np
import matplotlib.pyplot as plt


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
        yield pop_denorm[best_idx], fitness[best_idx]


result = list(
    de(lambda x: sum(x**2), bounds=[(-100, 100)] * 30, pop_size=100, gen_max=1500))
x, f = zip(*result)
plt.plot(np.log(f))
plt.show()
print(result[-1])
