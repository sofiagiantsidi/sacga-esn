import numpy as np
import pandas as pd
import random
import math
import time
from scipy.spatial.distance import pdist

class SACuckooGA_nobatches:
    def __init__(self, population, population_size, generations, levy_alpha, leaking_rate_range, spectral_radius_range,
                 omega_range, gamma_range, res_size_range, sparsity_res_range, sparsity_input_range, data, train_size, input_dim):
        # Initialization with function references
        self.population = population
        self.population_size = population_size
        self.generations = generations
        self.levy_alpha = levy_alpha
        self.leaking_rate_range = leaking_rate_range
        self.spectral_radius_range = leaking_rate_range
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
        beta = 1.5  # Levy distribution parameter

        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
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
                self.clip_value(c + step, range_)
                for c, step, range_ in zip(
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

            cuckoo_size = int(cuckoo[-3])
            modified_cuckoo = cuckoo[:-3] + (cuckoo_size,) + cuckoo[-2:]

            cuckoo_population.append(modified_cuckoo)

        return cuckoo_population

    def separation_ratio_func(self, reservoir_states_test_set, x_test, inf_large=1e12, inf_small=-1e12):
        inputs = np.asarray(x_test)
        reservoir_states = np.asarray(reservoir_states_test_set)
   
        input_distances = pdist(inputs, metric='euclidean')
        reservoir_distances = pdist(reservoir_states, metric='euclidean')
   
        with np.errstate(divide='ignore', invalid='ignore'):
            separation_ratios = np.divide(reservoir_distances, input_distances)
   
        separation_ratios[np.isposinf(separation_ratios)] = inf_large
        separation_ratios[np.isneginf(separation_ratios)] = inf_small

        return np.mean(separation_ratios)

    def run(self):
        print('SA-Cuckoo-GA: Error minimization (fitness score) & reservoir properties (separation ratio).')
       
        population = self.population
        population_size = self.population_size

        quantile10s = []
        averages = []

        for gen in range(self.generations):
            if gen > 0:
                population = new_population

            populations_fitness_scores = []

            cuckoo_population_size = 1   # You can change this as needed
            cuckoo_population = self.cuckoo_algorithm(cuckoo_population_size)

            population.extend(cuckoo_population)
            population_size = len(population)

            print(f'\n--------------------------------------Generation {gen}, Population size: {len(population)}')

            separation_ratios_means = []
            poor_inds = []
         
            for i, ind in enumerate(population):
                print('')
                print(f'\nIndividual {i}: {ind}')
                   
                self.RC_model = ReservoirComputingTorch(self.data, self.train_size, self.input_dim, ind)

                rmse, reservoir_states_test, X_test, Y_test, predictions = self.RC_model.ESN()
               
                start_time = time.time()
                separation_ratio = self.separation_ratio_func(
                    pd.DataFrame(reservoir_states_test),
                    X_test.reset_index(drop=True)
                )
                end_time = time.time()

                print(f'Time taken for Separation ratio computation: {end_time - start_time:.2f} seconds')
               
                ind_score_rmse = round(rmse, 6)
                ind_score_sep_ratio = round(separation_ratio, 6)

                print(f'Fitness Score (RMSE): {ind_score_rmse}')
                print('Fitness Score (Separatio Ratio):', {ind_score_sep_ratio})
               
                separation_ratios_means.append(ind_score_sep_ratio)
                populations_fitness_scores.append(ind_score_rmse)

                if ind_score_rmse < self.best_fitness_score:
                    self.best_fitness_score = ind_score_rmse
                    self.best_individual = ind

            average_fitness = np.mean(populations_fitness_scores)
            averages.append(average_fitness)

            print(f'Average Fitness: {average_fitness}')

            sorted_population = [ind for _, ind in sorted(zip(populations_fitness_scores, population))]

            quantile_10 = np.percentile(populations_fitness_scores, 10)
            quantile10s.append(quantile_10)

            print(f'Quantile 10 Fitness Score: {quantile_10}')
           
            elites = []
            mutation_rates = [None] * len(population)

            # First pass: Identify elites based on strict criteria
            for i in range(len(population)):
                if (
                    1 <= separation_ratios_means[i] <= 100 and
                    populations_fitness_scores[i] <= quantile_10
                ):
                    elites.append(i)
                    mutation_rates[i] = 0   # Assign mutation rate of 0 to elites
            
            # If not enough elites, select top 2 based on fitness(RMSE)
            if len(elites) < 2:
                print('Insufficient individuals meet the strict criteria; selecting top performers as elites.')

                sorted_indices = sorted(range(len(population)), key=lambda x: populations_fitness_scores[x])
                elites = sorted_indices[:2]

                for i in elites:
                    mutation_rates[i] = 0

            # Final check (should never fail, but kept as a safeguard)
            if len(elites) < 2:
                raise ValueError("Not enough elites found. Consider adjusting criteria or the population size.")

            # Apply mutation rates and crossover
            for i in range(len(population)):
                if mutation_rates[i] is None and populations_fitness_scores[i] > average_fitness:
                    mutation_rates[i] = 0.95
                else:
                    mutation_rates[i] = 0.1 if 1 < separation_ratios_means[i] < 100 else 0.8

            new_population = [population[i] for i in elites]

            print('No of elites populated:', len(elites))
                               
            for i in range(len(mutation_rates)):
                if i in elites:
                    continue
                           
                if random.random() < mutation_rates[i]:
                    mutated_child = list(population[i])

                    num_genes_to_mutate = random.randint(1, len(mutated_child))
                    genes_to_mutate = random.sample(range(len(mutated_child)), num_genes_to_mutate)

                    for gene_index in genes_to_mutate:
                        if gene_index == 4:
                            mutated_child[gene_index] = random.randint(*self.res_size_range)
                        else:
                            range_list = [
                                self.leaking_rate_range,
                                self.spectral_radius_range,
                                self.omega_range,
                                self.gamma_range,
                                None,
                                self.sparsity_res_range,
                                self.sparsity_input_range
                            ]
                            mutated_child[gene_index] = random.uniform(*range_list[gene_index])

                    new_population.append(tuple(mutated_child))

                else:    # crossover
                    p1 = random.choice(elites)
                    parent1 = population[p1]
                    parent2 = population[i]

                    child = tuple(
                        random.choice([parent1[j], parent2[j]])
                        for j in range(len(parent1))
                    )

                    new_population.append(child)
           
            worst_individual = max(range(len(population)), key=lambda x: populations_fitness_scores[x])
            print('worst ind:', worst_individual)
            # Remove the worst individual from the population to make room for the Cuckoo child
            new_population.pop(worst_individual)
               
            population = new_population

            print(f'End of Generation {gen}')
           
            if gen == 9:
                print(time.time())
                print(self.best_individual)
                print(self.best_fitness_score)
               
        print(f'Best Individual: {self.best_individual}')
        print(f'Best Fitness Score: {self.best_fitness_score}')
       
        return self.best_individual, self.best_fitness_score
