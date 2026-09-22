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
        
        fitness_scores.append(jaccard_score(ground_truth, weighted_pred))
    
    return np.mean(fitness_scores)

# Define your dataset and parameters
dataset_path = os.environ.get("ISIC_DATASET_PATH", "data/ISIC_Challenge_Dataset")
soft_predictions_dir = os.environ.get("SOFT_PREDICTIONS_DIR", "results/soft_predictions")
population_size = 100
num_generations = 50
mutation_rate = 0.1 
crossover_rate = 0.8

# Load validation data
_, (valid_x, valid_y), _ = load_data(dataset_path)

# Initialize the population
population = [normalize_weights(np.random.rand(5)) for _ in range(population_size)]

best_fitness_per_generation = []
average_fitness_per_generation = []
best_fitness_overall = 0
best_generation = 0

# Microbial GA
for generation in range(num_generations):
    print(f"Starting generation {generation + 1}")

    # Randomly select two individuals
    idx1, idx2 = random.sample(range(population_size), 2)
    individual1, individual2 = population[idx1], population[idx2]

    # Calculate their fitness
    fitness1 = calculate_fitness(individual1, valid_y, soft_predictions_dir)
    fitness2 = calculate_fitness(individual2, valid_y, soft_predictions_dir)
    
    # Competition: Determine winner and loser
    if fitness1 > fitness2:
        winner, loser = individual1, individual2
        loser_idx = idx2
    else:
        winner, loser = individual2, individual1
        loser_idx = idx1
    
    # Crossover: Winner influences the loser
    for i in range(len(loser)):
        if np.random.rand() < crossover_rate:
            loser[i] = winner[i]
    
    # Mutation
    loser = mutate(loser, mutation_rate)
    
    # Replace the original loser with the modified loser
    population[loser_idx] = loser
    
    # Track fitness statistics
    best_fitness = max(fitness1, fitness2)
    average_fitness = np.mean([calculate_fitness(individual, valid_y, soft_predictions_dir) for individual in population])

    best_fitness_per_generation.append(best_fitness)
    average_fitness_per_generation.append(average_fitness)

    # Track the best fitness and the corresponding generation
    if best_fitness > best_fitness_overall:
        best_fitness_overall = best_fitness
        best_generation = generation + 1

    print(f"Generation {generation + 1}: Best fitness = {best_fitness}, Average fitness = {average_fitness}")

print(f"Best fitness overall: {best_fitness_overall} achieved at generation {best_generation}")

# After all generations, save the best weights and fitness data
best_weights = population[best_fitness_per_generation.index(max(best_fitness_per_generation))]
np.save("best_weights_microbial.npy", best_weights)
np.save("best_fitness_per_generation_microbial.npy", best_fitness_per_generation)
np.save("average_fitness_per_generation_microbial.npy", average_fitness_per_generation)

print("Best weights and fitness data saved.")

# Plotting the fitness graph
plt.figure(figsize=(10, 6))
plt.plot(best_fitness_per_generation, label='Best Fitness', marker='o')
plt.plot(average_fitness_per_generation, label='Average Fitness', linestyle='--', marker='x')
plt.title('Best and Average Fitness per Generation (Microbial GA)')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()

# Load the best weights
best_weights = np.load("best_weights_microbial.npy")
print("Final ensemble weights:", best_weights)
print("Sum of weights:", np.sum(best_weights))