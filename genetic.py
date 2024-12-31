import random
import numpy as np
from scipy.linalg import eigh


class GeneticOptimizer:
    def __init__(self, population_size=70, generations=200, mutation_rate=0.1, elite_size=2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def initialize_population(self, n_total, n_sensors):
        """Initialize random population of sensor configurations"""
        population = []
        for _ in range(self.population_size):
            # Create a random selection of sensor positions
            chromosome = np.zeros(n_total, dtype=bool)
            sensor_positions = np.random.choice(n_total, n_sensors, replace=False)
            chromosome[sensor_positions] = True
            population.append(chromosome)
        return population

    def fitness_efi(self, chromosome, mode_matrix):
        """Calculate fitness using EFI methodology"""
        selected_modes = mode_matrix[chromosome]
        fim = selected_modes @ selected_modes.T
        eigenvals = eigh(fim, eigvals_only=True)
        # Use determinant of FIM as fitness measure
        return np.sum(np.log(eigenvals[eigenvals > 1e-10]))

    def fitness_efi_dpr(self, chromosome, mode_matrix, frequencies):
        """Calculate fitness using EFI-DPR methodology"""
        selected_modes = mode_matrix[chromosome]
        
        # Calculate DPR component
        dpr = np.zeros(selected_modes.shape[0])
        for i in range(selected_modes.shape[1]):
            dpr += selected_modes[:, i]**2 / frequencies[i]
        dpr_normalized = dpr / np.max(dpr)
        
        # Calculate EFI component
        fim = selected_modes @ selected_modes.T
        eigenvals = eigh(fim, eigvals_only=True)
        efi_score = np.sum(np.log(eigenvals[eigenvals > 1e-10]))
        
        # Combine EFI and DPR scores
        return efi_score * np.mean(dpr_normalized)

    def select_parents(self, population, fitness_scores):
        """Select parents using tournament selection"""
        tournament_size = 3
        parents = []
        for _ in range(len(population) - self.elite_size):
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        return parents

    def crossover(self, parent1, parent2):
        """Perform crossover while maintaining the same number of sensors"""
        n_sensors = np.sum(parent1)
        child = np.zeros_like(parent1)
        
        # Get indices where sensors are placed in both parents
        parent1_indices = np.where(parent1)[0]
        parent2_indices = np.where(parent2)[0]
        
        # Randomly select crossover point
        crossover_point = random.randint(0, n_sensors)
        
        # Take sensors from both parents
        selected_indices = np.concatenate([
            parent1_indices[:crossover_point],
            parent2_indices[crossover_point:]])
        
        # If we have too many or too few sensors, adjust randomly
        while len(selected_indices) != n_sensors:
            if len(selected_indices) > n_sensors:
                selected_indices = np.random.choice(
                    selected_indices, n_sensors, replace=False)
            else:
                available = np.setdiff1d(np.arange(len(parent1)), selected_indices)
                additional = np.random.choice(
                    available, n_sensors - len(selected_indices), replace=False)
                selected_indices = np.concatenate([selected_indices, additional])
        
        child[selected_indices] = True
        return child

    def mutate(self, chromosome):
        """Perform mutation while maintaining the same number of sensors"""
        if random.random() < self.mutation_rate:
            n_sensors = np.sum(chromosome)
            sensor_positions = np.where(chromosome)[0]
            non_sensor_positions = np.where(~chromosome)[0]
            
            # Randomly select one sensor to move
            sensor_to_move = np.random.choice(sensor_positions)
            new_position = np.random.choice(non_sensor_positions)
            
            # Move the sensor
            chromosome[sensor_to_move] = False
            chromosome[new_position] = True
        
        return chromosome