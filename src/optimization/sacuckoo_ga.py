import random
import math
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from src.models.esn_torch import ReservoirComputingTorch


class SACuckooGA_nobatches:
    def __init__(
        self,
        population,
        population_size,
        generations,
        levy_alpha,
        leaking_rate_range,
        spectral_radius_range,
        omega_range,
        gamma_range,
        res_size_range,
        sparsity_res_range,
        sparsity_input_range,
        data,
        train_size,
        input_dim
    ):
        self.population = population
        self.population_size = population_size
        self.generations = generations
        self.levy_alpha = levy_alpha

        self.leaking_rate_range = leaking_rate_range
        self.spectral_radius_range = spectral_radius_range
        self.omega_range = omega_range
        self.gamma_range = gamma_range
        self.res_size_range = res_size_range
        self.sparsity_res_range = sparsity_res_range
        self.sparsity_input_range = sparsity_input_range

        self.data = data
        self.train_size = train_size
        self.input_dim = input_dim

        self.RC_model = None
        self.best_individual = None
        self.best_fitness_score = float('inf')

    def levy_flight(self):
        beta = 1.5

        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)

        step = u / abs(v) ** (1 / beta)
        return step / self.levy_alpha

    def clip_value(self, value, value_range):
        return max(value_range[0], min(value, value_range[1]))

    def cuckoo_algorithm(self, cuckoo_population_size):
        cuckoo_population = []

        for _ in range(cuckoo_population_size):
            cuckoo = (
                random.uniform(*self.leaking_rate_range),
                random.uniform(*self.spectral_radius_range),
                random.uniform(*self.omega_range),
                random.uniform(*self.gamma_range),
                random.randint(*self.res_size_range),
                random.uniform(*self.sparsity_res_range),
                random.uniform(*self.sparsity_input_range)
            )

            levy_step = [self.levy_flight() * random.choice([-1, 1]) for _ in range(len(cuckoo))]

            cuckoo = tuple(
                self.clip_value(c + step, rng)
                for c, step, rng in zip(
                    cuckoo,
                    levy_step,
                    [
                        self.leaking_rate_range,
                        self.spectral_radius_range,
                        self.omega_range,
                        self.gamma_range,
                        self.res_size_range,
                        self.sparsity_res_range,
                        self.sparsity_input_range
                    ]
                )
            )

            cuckoo_size = int(cuckoo[4])
            modified_cuckoo = cuckoo[:4] + (cuckoo_size,) + cuckoo[5:]
            cuckoo_population.append(modified_cuckoo)

        return cuckoo_population

    def separation_ratio_func(self, reservoir_states, x_test, inf_large=1e12, inf_small=-1e12):
        inputs = np.asarray(x_test)
        reservoir_states = np.asarray(reservoir_states)

        input_distances = pdist(inputs, metric='euclidean')
        reservoir_distances = pdist(reservoir_states, metric='euclidean')

        with np.errstate(divide='ignore', invalid='ignore'):
            separation_ratios = np.divide(reservoir_distances, input_distances)

        separation_ratios[np.isposinf(separation_ratios)] = inf_large
        separation_ratios[np.isneginf(separation_ratios)] = inf_small

        return np.mean(separation_ratios)

    def run(self):
        population = self.population

        for gen in range(self.generations):
            if gen > 0:
                population = new_population

            cuckoo_population = self.cuckoo_algorithm(1)
            population.extend(cuckoo_population)

            populations_fitness_scores = []
            separation_ratios_means = []

            print(f"\nGeneration {gen}, Population size: {len(population)}")

            for i, ind in enumerate(population):
                print(f"\nIndividual {i}: {ind}")

                self.RC_model = ReservoirComputingTorch(
                    self.data, self.train_size, self.input_dim, ind
                )

                rmse, reservoir_states_test, X_test, Y_test, predictions = self.RC_model.ESN()

                separation_ratio = self.separation_ratio_func(
                    pd.DataFrame(reservoir_states_test),
                    X_test.reset_index(drop=True)
                )

                populations_fitness_scores.append(rmse)
                separation_ratios_means.append(separation_ratio)

                if rmse < self.best_fitness_score:
                    self.best_fitness_score = rmse
                    self.best_individual = ind

            average_fitness = np.mean(populations_fitness_scores)
            print(f"Average Fitness: {average_fitness}")

            sorted_population = [ind for _, ind in sorted(zip(populations_fitness_scores, population))]

            quantile_10 = np.percentile(populations_fitness_scores, 10)

            elites = []
            for i in range(len(population)):
                if separation_ratios_means[i] > 1 and populations_fitness_scores[i] <= quantile_10:
                    elites.append(i)

            if len(elites) < 2:
                elites = sorted(range(len(population)), key=lambda x: populations_fitness_scores[x])[:2]

            new_population = [population[i] for i in elites]

            for i in range(len(population)):
                if i in elites:
                    continue

                if random.random() < 0.5:
                    parent = population[random.choice(elites)]
                    mutated = list(parent)

                    gene = random.randint(0, len(mutated) - 1)
                    mutated[gene] = random.uniform(0, 1)

                    new_population.append(tuple(mutated))
                else:
                    new_population.append(population[i])

            population = new_population

        print("Best Individual:", self.best_individual)
        print("Best Fitness:", self.best_fitness_score)

        return self.best_individual, self.best_fitness_score
