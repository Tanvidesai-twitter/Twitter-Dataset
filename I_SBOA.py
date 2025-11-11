# Improved Secretary Bird Optimization Algorithm
import numpy as np
from save_load import load
from Detection import MLRK


def fit_func(x, learn_rate):
    x_train = load(f'x_train_{learn_rate}')
    x_test= load(f'x_test_{learn_rate}')
    y_train = load(f'y_train_{learn_rate}')
    y_test = load(f'y_test_{learn_rate}')

    soln = np.round(x)
    selected_indices = np.where(soln == 1)[0]
    if len(selected_indices) == 0:
        selected_indices = np.where(soln == 0)[0]
        if len(selected_indices) == 0:
            selected_indices = np.random.randint(0, x_test.shape[1], x_test.shape[1]-10)

    x_train = x_train[:, selected_indices]
    x_test = x_test[:, selected_indices]

    pred, met = MLRK(x_train, y_train, x_test, y_test)

    fit = 1 / met[0]

    return fit


def I_SBOA(Objective_func, lb, ub, pop_size, prob_size, epochs, learn_rate):

    population = np.random.uniform(lb, ub, size=(pop_size, prob_size))
    lb = np.array(lb)
    ub = np.array(ub)

    best_solution = None
    best_fitness = float('inf')

    for i in range(epochs):
        for j in range(pop_size):

            population[j, population[j] < lb] = lb[population[j] < lb]
            population[j, population[j] > ub] = ub[population[j] > ub]

            fitness = Objective_func(population[j], learn_rate)

            if fitness < best_fitness:
                best_solution = population[j]
                best_fitness = fitness

            # Hunting strategy of secretary bird (exploration phase)
            R = np.random.uniform(0, 1)
            RB = np.random.randn(1, prob_size)

            # Stage 1 (Searching for Prey)
            if i < (1/3)*epochs:
                new_position = population[j] + (np.random.choice(population[j]) - np.random.choice(population[j])) * R

            # Stage 2 (Consuming Prey)
            elif (1/3)*epochs < i < (2/3)*epochs:
                new_position = best_solution + np.exp((i+1/epochs)**4) * (RB - 0.5) * (best_solution - population[j])

            # Stage 3 (Attacking Prey)
            elif i > (2/3)*epochs:
                # Hybrid - Velocity
                x_a = np.random.rand(prob_size)
                r1, r2, r3 = np.random.rand(3) # random number b/w 0 to 1
                U2 = np.random.choice([0, 1])
                Ms = 1.0
                mi = 0.5
                F = 6.67430e-11  # Universal gravitational constant
                epsilon = 1e-10
                Ti = np.random.normal()  # Random value from normal distribution
                R_min, R_max = 0, 1
                R_it = np.linalg.norm(j - np.random.rand())
                R_i_norm = (R_it - R_min) / (R_max - R_min)
                a_it = (F * (Ms + mi) * Ti**2 / (4 * np.pi**2))**(1/3)
                H = Ms + mi * (F - 1) * (a_it + epsilon) + 2 * (R_it + epsilon)
                Vit = x_a - population[j] * r1 * H + F * U2 * (1-R_i_norm) * r2 * (ub - population[j]) * lb * r3
                new_position = best_solution + ((1 - ((i+1)/epochs))**(2*((i+1)/epochs))) * population[j] * Vit

            if Objective_func(new_position, learn_rate) < fitness:
                population[j] = new_position

            # Escape strategy of secretary bird (exploitation stage)
            #  C1 = Camouflage by environment
            #  C2 = Fly or run away
            r = 0.5
            R2 = np.random.uniform(prob_size)
            X_random = np.random.choice(population[j])
            K = np.random.choice([1, 2])

            if r < R:
                new_position = best_solution + (2*RB-1)*(1-((i+1)/epochs))**2 * population[j] # C1
            else:
                new_position = population[j] + R2 * (X_random - K * population[j]) # C2

            if Objective_func(new_position, learn_rate) < fitness:
                population[j] = new_position

    return best_solution, best_fitness

