import numpy as np
import random
import math
import time
import pandas as pd
from scipy.spatial.distance import pdist

from src.models.esn_torch import ReservoirComputingTorch


class SACuckooGA_nobatches:
    def __init__(self, population, population_size, generations, levy_alpha,
                 leaking_rate_range, spectral_radius_range,
                 omega_range, gamma_range, res_size_range,
                 sparsity_res_range, sparsity_input_range,
                 data, train_size, input_dim):

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

        self.best_individual = None
        self.best_fitness_score = float('inf')

    def levy_flight(self):
        beta = 1.5
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)

        step = u / abs(v) ** (1 / beta)
        return step / self.levy_alpha

    def separation_ratio_func(self, reservoir_states, x_test):
        inputs = np.asarray(x_test)
        reservoir_states = np.asarray(reservoir_states)

        input_distances = pdist(inputs, metric='euclidean')
        reservoir_distances = pdist(reservoir_states, metric='euclidean')

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.divide(reservoir_distances, input_distances)

        ratios[np.isinf(ratios)] = 1e12
        return np.mean(ratios)

    def run(self):
        population = self.population

        for gen in range(self.generations):
            print(f"\nGeneration {gen}")

            fitness_scores = []

            for ind in population:
                model = ReservoirComputingTorch(
                    self.data, self.train_size, self.input_dim, ind
                )

                rmse, res_states, X_test, Y_test, preds = model.ESN()

                fitness_scores.append(rmse)

                if rmse < self.best_fitness_score:
                    self.best_fitness_score = rmse
                    self.best_individual = ind

        return self.best_individual, self.best_fitness_score