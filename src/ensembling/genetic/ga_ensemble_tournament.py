import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import random
import cv2
from sklearn.metrics import jaccard_score
from common.data import load_data, create_dir
from common.soft_predictions import load_soft_predictions, load_and_process_mask
import matplotlib.pyplot as plt
import tensorflow as tf

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Using GPU: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, using CPU.")

# Set seed for reproducibility
np.random.seed(42)

def normalize_weights(weights):
    """Normalize the weights to ensure they sum to 1."""
    return weights / np.sum(weights)

def mutate(weights, mutation_rate):
    """Apply mutation to the weights."""
    mutation = np.random.randn(len(weights)) * mutation_rate
    weights += mutation
    weights = np.clip(weights, 0, 1)
    return normalize_weights(weights)

def crossover(parent1, parent2, crossover_rate):
    """Apply crossover between two parents."""
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return normalize_weights(child1), normalize_weights(child2)
    return parent1, parent2

def tournament_selection(population, fitness, k=3):
    """Select an individual from the population using tournament selection."""
    selected = random.sample(list(zip(population, fitness)), k)
    return max(selected, key=lambda x: x[1])[0]

def calculate_fitness(weights, ground_truth_paths, soft_predictions_dir):
    fitness_scores = []
    for i in range(len(ground_truth_paths)):
        soft_predictions = load_soft_predictions(soft_predictions_dir, i)
        weighted_pred = np.tensordot(weights, soft_predictions, axes=(0, 0))
        
        # Threshold to binary
        weighted_pred = (weighted_pred > 0.5).astype(np.float32).flatten()
        
        # Load and process ground truth
        ground_truth = load_and_process_mask(ground_truth_paths[i])
        ground_truth = (ground_truth > 0.5).astype(np.float32).flatten()  # Ensure ground truth is binary
        
        # Ensure both arrays are of the same length
        if len(weighted_pred) != len(ground_truth):
            raise ValueError(f"Inconsistent length: {len(weighted_pred)} in prediction vs {len(ground_truth)} in ground truth.")
        
        fitness_scores.append(jaccard_score(ground_truth, weighted_pred))
    
    return np.mean(fitness_scores)

# Define your dataset and parameters

if __name__ == "__main__":
    dataset_path = os.environ.get("ISIC_DATASET_PATH", "data/ISIC_Challenge_Dataset")
    soft_predictions_dir = os.environ.get("SOFT_PREDICTIONS_DIR", "results/soft_predictions")
    population_size = 100
    num_generations = 50
    initial_crossover_rate = 0.7
    initial_mutation_rate = 0.05
    elitism_rate = 0.05

    # Load validation data
    _, (valid_x, valid_y), _ = load_data(dataset_path)

    # Initialize the population
    population = [normalize_weights(np.random.rand(5)) for _ in range(population_size)]

    best_fitness_per_generation = []
    average_fitness_per_generation = []
    best_fitness_overall = 0
    best_generation = 0

    for generation in range(num_generations):
        print(f"Starting generation {generation + 1}")
    
        # Calculate fitness for each individual in the population
        fitness = []
        for individual in population:
            fitness.append(calculate_fitness(individual, valid_y, soft_predictions_dir))
    
        best_fitness = np.max(fitness)
        average_fitness = np.mean(fitness)
    
        # Track the best fitness and the corresponding generation
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_generation = generation + 1

        best_fitness_per_generation.append(best_fitness)
        average_fitness_per_generation.append(average_fitness)
    
        print(f"Generation {generation + 1}: Best fitness = {best_fitness}, Average fitness = {average_fitness}")
    
        # Sort population by fitness (descending)
        sorted_population = [population[i] for i in np.argsort(fitness)[::-1]]
    
        # Elitism: Carry over the top individuals unchanged
        num_elites = int(elitism_rate * population_size)
        new_population = sorted_population[:num_elites]
    
        # Adjust crossover and mutation rates dynamically
        crossover_rate = initial_crossover_rate * (1 - generation / num_generations)
        mutation_rate = initial_mutation_rate * (1 + generation / num_generations)
    
        # Perform tournament selection, crossover, and mutation on the rest
        while len(new_population) < population_size:
            parent1 = tournament_selection(sorted_population, fitness)
            parent2 = tournament_selection(sorted_population, fitness)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
        
            # Apply mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
        
            new_population.append(child1)
            new_population.append(child2)
    
        # Replace the old population with the new one
        population = new_population[:population_size]

    # After all generations, save the best weights and fitness data
    best_weights = sorted_population[0]
    np.save("best_weights_tournament.npy", best_weights)
    np.save("best_fitness_per_generation_tournament.npy", best_fitness_per_generation)
    np.save("average_fitness_per_generation_tournament.npy", average_fitness_per_generation)

    print(f"Best fitness overall: {best_fitness_overall} achieved at generation {best_generation}")
    print("Best weights and fitness data saved.")

    # Plotting the fitness graph
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_per_generation, label='Best Fitness', marker='o')
    plt.plot(average_fitness_per_generation, label='Average Fitness', linestyle='--', marker='x')
    plt.title('Best and Average Fitness per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()
