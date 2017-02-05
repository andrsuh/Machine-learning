import numpy as np


class DiffEvolution:
    def __init__(
            self, tfunc, dimension, popsize=50,
            mutation_prob=0.9, niter=1000):
        self.tfunc = tfunc
        self.dimension = dimension
        self.popsize = popsize
        self.niter = niter
        self.mutation_prob = mutation_prob

    def generate_population(self):
        self.population = []
        for i in range(self.popsize):
            sample = np.random.randint(0, 2, self.dimension)
            self.population.append((sample, self.tfunc(np.array(sample) == 1)))

    def get_3sample_exclude(self, exclude):
        for i in range(3):
            while True:
                n = np.random.randint(len(self.population))
                if n not in exclude:
                    yield self.population[n][0]
                    exclude.append(n)
                    break

    def fit(self):
        self.generate_population()
        for it in range(self.niter):
            print(it)
            new_population = []
            for i, (sample, quality) in enumerate(self.population):
                a, b, c = self.get_3sample_exclude([i])
                exactly_mutated = np.random.randint(self.dimension)
                new_sample = np.array(sample)
                for j, gene in enumerate(sample):
                    if np.random.rand() < self.mutation_prob or j == exactly_mutated:
                        new_sample[j] = a[j] and (b[j] or c[j])
                new_quality = self.tfunc(np.array(new_sample) == 1)
                if quality < new_quality:
                    new_population.append((new_sample, new_quality))
                else:
                    new_population.append((sample, quality))
            self.population = new_population

        return self.population
