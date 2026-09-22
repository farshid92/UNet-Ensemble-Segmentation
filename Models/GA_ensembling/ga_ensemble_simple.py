import numpy as np
import os
import random
import cv2
from sklearn.metrics import jaccard_score
from train import load_data
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

def load_and_process_mask(mask_path):
    """Loads and processes the mask from the given path."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))  # Ensure it matches the input size
    mask = mask / 255.0  # Normalize to [0, 1]
    return mask

def load_soft_predictions(soft_predictions_dir, index):
    """Loads the soft predictions for a specific image."""
    models = ['unet', 'convnextbase', 'mobilenetv3large', 'resnet50v2', 'vgg19']
    return np.array([np.load(os.path.join(soft_predictions_dir, f"{model}_preds_{index}.npy")) for model in models])

def normalize_weights(weights):
    """Normalize the weights to ensure they sum to 1."""
    return weights / np.sum(weights)

def mutate(weights, mutation_rate):
    """Apply mutation to the weights."""
    mutation = np.random.randn(len(weights)) * mutation_rate
    weights += mutation
    weights = np.clip(weights, 0, 1)
    return normalize_weights(weights)

def crossover(parent1, parent2, crossover_rate=0.8):
    """Apply crossover between two parents."""
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return normalize_weights(child1), normalize_weights(child2)
    return parent1, parent2

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
dataset_path = os.environ.get("ISIC_DATASET_PATH", "data/ISIC_Challenge_Dataset")
soft_predictions_dir = os.environ.get("SOFT_PREDICTIONS_DIR", "results/soft_predictions")
population_size = 100
num_generations = 100
crossover_rate = 0.8
initial_mutation_rate = 0.05  # Reduced mutation rate
elitism_rate = 0.05  # Elitism rate (top 5% of individuals carried over)

# Load validation data
_, (valid_x, valid_y), _ = load_data(dataset_path)

# Initialize the population
population = [normalize_weights(np.random.rand(5)) for _ in range(population_size)]

best_fitness_per_generation = []
average_fitness_per_generation = []

for generation in range(num_generations):
    print(f"Starting generation {generation + 1}")
    
    # Calculate fitness for each individual in the population
    fitness = []
    for individual in population:
        fitness.append(calculate_fitness(individual, valid_y, soft_predictions_dir))
    
    best_fitness = np.max(fitness)
    average_fitness = np.mean(fitness)
    
    best_fitness_per_generation.append(best_fitness)
    average_fitness_per_generation.append(average_fitness)
    
    print(f"Generation {generation + 1}: Best fitness = {best_fitness}")
    
    # Sort population by fitness (descending)
    sorted_population = [population[i] for i in np.argsort(fitness)[::-1]]
    
    # Elitism: Carry over the top individuals unchanged
    num_elites = int(elitism_rate * population_size)
    new_population = sorted_population[:num_elites]
    
    # Perform crossover and mutation on the rest
    while len(new_population) < population_size:
        parent1, parent2 = random.choices(sorted_population[:50], k=2)
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        
        # Apply mutation with decreasing mutation rate
        mutation_rate = initial_mutation_rate * (1 - generation / num_generations)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        
        new_population.append(child1)
        new_population.append(child2)
    
    # Replace the old population with the new one
    population = new_population[:population_size]

# After all generations, save the best weights and fitness data
best_weights = sorted_population[0]
np.save("best_weights_genetic_algorithm_ensemble_100gen_simple.npy", best_weights)
np.save("best_fitness_per_generation_genetic_algorithm_ensemble_100gen_simple.npy", best_fitness_per_generation)
np.save("average_fitness_per_generation_genetic_algorithm_ensemble_100gen_simple.npy", average_fitness_per_generation)

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

# Load the best weights
# best_weights = np.load("best_weights_genetic_algorithm_ensemble_100gen_simple.npy")
# print("Final ensemble weights:", best_weights)
# print("Sum of weights:", np.sum(best_weights))