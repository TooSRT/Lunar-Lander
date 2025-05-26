import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# NN model, lunar lander environment, run the model
def run(model, env, n_sim):
    model.eval()
    
    all_rewards = np.zeros(n_sim)
    with torch.no_grad():
        for idx_sim in range(n_sim):
            observation, info = env.reset() # Reset environment
            terminated = False
            truncated = False
            while not terminated and not truncated:
                observation = torch.from_numpy(observation)
                action = torch.argmax(model(observation))
                #action = torch.argmax(torch.rand(4)) #generate random action (0,1,2,3)
                observation, reward, terminated, truncated, info = env.step(action.item())
                all_rewards[idx_sim] += reward
                
    return np.max(all_rewards)

#Create the lunar lander ANN model
class LunarLanderANN(nn.Module):
    def __init__(self):
        super(LunarLanderANN, self).__init__()
        
        self.input_dim = 8    # [x, y, vx, vy, theta, omega, leg1, leg2]
        self.output_dim = 4   # [0,1,2,3] actions
        
        self.hdim_1 = 8  #neurons in fc1 layers
        
        self.fc1 = nn.Linear(self.input_dim, self.hdim_1)
        torch.nn.init.uniform_(self.fc1.weight, a=-1.0, b=1.0)
        self.activation1 = nn.ReLU()
        
        self.fc2 = nn.Linear(self.hdim_1, self.output_dim)
        torch.nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.fc2(out)
        return out

def tournament(population, fitness_scores, tsize):
    """
    Select the individual with the best rewards (score) for landing
    
    Args:
        population (list): List of our  population (instances of LunarLanderANN)
        fitness_scores (np.array): List of the scores of our instances
        tsize (int): Size of the tournament (number of instances choosen aleatory)
    
    Returns:
        list: New population (instances of LunarLanderANN) selected
    """
    selected_population = population.copy() #copy all the instances
    
    for i in range(len(population)):
        # select tsize individuals without replacement
        idx = np.random.choice(len(population), tsize, replace=False)
        #Do the tournament
        #select the index of the best individual of a tournament        
        best = idx[np.argmax(np.array(fitness_scores)[idx])] #here we want the highest score so argmax
        selected_population[i] = population[best]
    
    return selected_population

#tournament(population, fitness_scores, 10)

import copy
def get_param_vector(ann):
    """
    Extracts and concatenates the weights of fc1 and fc2 from the model into a single vector
    
    Args:
        ann (nn.module): An instance of LunarLanderANN
        
    Returns:
        param_vector (np.array): Concatenated vector of weights from fc1 and fc2
    """    
    #Extract and flatten fc1 weights
    fc1_weights = ann.fc1.weight.data.cpu().numpy().flatten()
    #Extract and flatten fc2 weights
    fc2_weights = ann.fc2.weight.data.cpu().numpy().flatten()
    #Concatenate the two parts
    return np.concatenate((fc1_weights, fc2_weights))

def get_reward(ann, env, param, n_sim):
    with torch.no_grad():
        numel1 = torch.numel(ann.fc1.weight)
        ann.fc1.weight.copy_(torch.from_numpy(param[:numel1].astype(np.float32)).view_as(ann.fc1.weight))
        ann.fc2.weight.copy_(torch.from_numpy(param[numel1:].astype(np.float32)).view_as(ann.fc2.weight))
        
    
def crossover(population):
    """
    Performs crossover on a population of individuals (LunarLanderANN models)
    
    For each pair of parents a random single crossover point is chosen for each parameter in the flattened parameter tensor. 
    Child 1 receives the first segment from parent 1 and the remaining segment from parent 2 
    while child 2 receives the first segment from parent 2 and the remaining segment from parent 1
    
    Args:
        population (list): List of LunarLanderANN instances 
        
    Returns:
        children (list): New list of individuals (children) resulting from the crossover
    """
    children = []
    div = int(len(population) / 2)  # be sure that pop_size is even, we divide the population in 2
    #print(population)
    #For each pair of parents (first half paired with second half)
    for i in range(div):
        #Parents that are gonna give their "weights"
        parent1 = population[i] #instances
        parent2 = population[i + div] #second part of instances
            
        param_vector1 = get_param_vector(parent1)
        param_vector2 = get_param_vector(parent2)

        cut = np.random.randint(1,len(param_vector1))

        #create the new weights for child1 and child2 from both of the parents
        child_param_vector1 = np.concatenate((param_vector1[:cut], param_vector2[cut:]))
        child_param_vector2 = np.concatenate((param_vector2[:cut], param_vector1[cut:]))
            
        #Create copies to form children
        #use deepcopy to copy a pytorch instance (then modify the weights) so we don't modify parents
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        #Update the weights of children with the new parameter vector
        get_reward(child1, env, child_param_vector1, n_sim)
        get_reward(child2, env, child_param_vector2, n_sim)
            
        children.append(child1)
        children.append(child2)

    return children

#crossover(population)

def mutate_individual(individual, mutation_rate, mutation_strength):
    """
    Mutates an individual's weights by adding small random noise
    
    For each parameter (weight tensor) in the network, a mutation mask is generated 
    where each element has a probability 'mutation_rate' to be mutated. The mutation 
    is performed by adding noise drawn from a normal distribution (mean=0, std=mutation_strength)
    to the selected weights
    
    Args:
        individual (nn.Module): Neural network model (i.e an instance of LunarLanderANN)
        mutation_rate (float): Probability of mutating each individual weight
        mutation_strength (float): Standard deviation of the normal distribution for the noise
    """
    with torch.no_grad(): 
        for param in individual.parameters():
            #choose a probability to have a mutation for each weight
            indv_mutate = torch.rand_like(param) < mutation_rate
            #Add noise based on normal gaussian distribution
            param.data.add_(indv_mutate.float() * torch.randn_like(param) * mutation_strength)
            #assure that weights stay between [-1,1]
            param.data.clamp_(-1, 1)

def genetic_algorithm(iterations, pop_size, tsize, mutation_rate, mutation_strength, n_sim, env):
    """
    Genetic algorithm to optimize LunaLanderANN
    
    Args:
        iterations (int): number of iterations for our algorthm (number of time we create new instances)
        pop_size (int): number of instances (LunarLanderANN)
        
    Returns:
    

    """
    #Initialize our instances
    population = [LunarLanderANN() for i in range(pop_size)]

    #store the best fitness and the best models
    best_fitness_history = []
    best_model_history = []
    
    for iteration in range(iterations):
        #evaluate the fitness (score)
        fitness_scores = [run(individual, env, n_sim) for individual in population]
        #print(fitness_scores)
        
        #save the best score
        best_idx = np.argmax(fitness_scores) #get index of the best fitness
        #print('best idx',best_idx)
        best_fitness = fitness_scores[best_idx] #access the best fitness
        best_model = copy.deepcopy(population[best_idx]) #get the best model of the population based on the best fitness

        #save the best model/fitness score
        best_fitness_history.append(best_fitness)
        best_model_history.append(best_model)
        
        print(f"Iteration {iteration}: best score (fitness) = {best_fitness:.2f}")
        
        #Selection of the best instances with the tournameny
        population = tournament(population, fitness_scores, tsize)
        
        #Crossover to get the best population
        population = crossover(population)
        
        #Mutate each individual in the new population
        for individual in population:
            mutate_individual(individual, mutation_rate, mutation_strength)
    
    df = pd.DataFrame({"Iteration": range(1, iterations + 1), "Best_Fitness": best_fitness_history})

    # Save to CSV file
    df.to_csv("best_scores.csv", index=False)
    
    return best_model_history, best_fitness_history

#----------test---------
env = gym.make(
    "LunarLander-v3",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    #render_mode="human", # None for intensive simulation
)

iterations = 400 #pop_size==iterations ? 400-500
pop_size = 60    #prendre 50-80 
n_sim = 10 # 10
tsize = 3               
mutation_rate = 0.5  
mutation_strength = 0.05
            
import time

#Mesure time
start_time = time.time()

best_model, best_scores = genetic_algorithm(iterations, pop_size, tsize, mutation_rate, mutation_strength, n_sim, env)

end_time = time.time()

total_time = end_time - start_time
print(f"Temps total de simulation : {total_time:.2f} secondes")
with open("simulation_time.txt", "w") as f:
    f.write(f"Temps total de simulation : {total_time:.2f} secondes\n")

#Create a random landing video based on the LunarLanderANN model and save it 
from gymnasium.wrappers import RecordVideo
if best_model:
    best_individual = best_model[-1]  #


    env = gym.make(
        "LunarLander-v3",  
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode="rgb_array",  # Important pour l'enregistrement vidéo
    )


    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda t: True, video_length=500, disable_logger=True)
    run(best_individual, env, 5)
    env.close()  
    print("Videos saved")

else:
    print("No model found")

