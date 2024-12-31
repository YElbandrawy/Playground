import random
import numpy as np
from base import SensorOptimizer
from genetic import GeneticOptimizer
from scipy.linalg import eigh

class SensorOptimizer(SensorOptimizer):  # Inherits from existing SensorOptimizer
    def genetic_optimization(self, method='EFI'):
        """
        Perform genetic algorithm optimization for sensor placement.
        
        Args:
            method (str): 'EFI' or 'EFI-DPR'
        
        Returns:
            tuple: (selected_indices, final_contributions)
        """    
        ga = GeneticOptimizer()
        n_total = len(self.nodes)
        
        # Initialize population
        population = ga.initialize_population(n_total, self.target_sensors)
        
        best_fitness = float('-inf')
        best_chromosome = None
        
        # Evolution loop
        for generation in range(ga.generations):
            # Calculate fitness for each chromosome
            fitness_scores = []
            for chromosome in population:
                if method == 'EFI':
                    fitness = ga.fitness_efi(chromosome, self.Main_Mat)
                else:  # EFI-DPR
                    fitness = ga.fitness_efi_dpr(chromosome, self.Main_Mat, self.modal_frequencies)
                fitness_scores.append(fitness)
            
            # Track best solution
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_chromosome = population[np.argmax(fitness_scores)]
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
            
            # Elitism: keep best solutions
            elite_indices = np.argsort(fitness_scores)[-ga.elite_size:]
            elite = [population[i] for i in elite_indices]
            
            # Select parents and create new population
            parents = ga.select_parents(population, fitness_scores)
            new_population = elite.copy()
            
            # Create offspring
            while len(new_population) < ga.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = ga.crossover(parent1, parent2)
                child = ga.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Get final selected indices and calculate contributions
        selected_indices = np.where(best_chromosome)[0]
        
        # Calculate final contributions based on method
        if method == 'EFI':
            selected_modes = self.Main_Mat[selected_indices]
            fim = selected_modes @ selected_modes.T
            _, eigenvects = eigh(fim)
            contributions = np.sum(eigenvects**2, axis=1)
        else:  # EFI-DPR
            selected_modes = self.Main_Mat[selected_indices]
            fim = selected_modes @ selected_modes.T
            _, eigenvects = eigh(fim)
            dpr = self.calculate_dpr(selected_modes)
            dpr_normalized = dpr / np.max(dpr)
            contributions = np.sum(eigenvects**2, axis=1) * dpr_normalized
        
        return selected_indices, contributions

    def optimize_positions_genetic(self, method='EFI'):
        """
        Execute the complete optimization process using genetic algorithm.
        
        Args:
            method (str): 'EFI' or 'EFI-DPR'
            
        Returns:
            dict: Results containing selected positions, coordinates, and contributions
        """
        print(f"\nStarting genetic algorithm optimization process ({method})...")
        try:
            # Read and process input data if not already done
            if self.nodes is None:
                self.read_coordinates()
            if self.Main_Mat is None:
                self.prepare_displacement_data()
            
            # Plot initial positions
            self.plot_nodes(self.nodes, "Initial Node Positions")
            
            # Run genetic optimization
            selected_indices, contributions = self.genetic_optimization(method)
            
            # Store results
            results = {
                'POS': self.POS[selected_indices],
                'COO': self.nodes[selected_indices],
                'Ed': contributions
            }
            
            # Plot final positions
            self.plot_nodes(self.nodes, f"Selected Sensor Positions (GA-{method})", selected_indices)
            
            # Save results
            self.save_results(results, suffix=f'_GA_{method}')
            
            print(f"\nGenetic algorithm optimization ({method}) completed successfully!")
            print("\nSelected sensor positions:")
            for i, (pos, coord) in enumerate(zip(results['POS'], results['COO']), 1):
                print(f"Sensor {i}: Node {pos} at coordinates ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
            
            return results
            
        except Exception as e:
            print(f"Error during genetic optimization: {str(e)}")
            raise