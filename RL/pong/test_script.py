import json
import numpy as np
from Agent import Agent

def train_and_evaluate(gamma, epsilon_decay, alpha, discretization_factor, episodes=350):

    agent = Agent(
        id=0,
        actions_n=3,
        obs_space_shape=(6,),
        gamma=gamma,
        epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=epsilon_decay,
        alpha=alpha
    )
    
    total_wins = 0
    total_losses = 0
    
    for episode in range(episodes):
        agent.decay_epsilon()
    
    performance = total_wins - total_losses
    return performance

gammas = [0.9, 0.95, 0.99]
epsilon_decays = [0.995, 0.9975, 0.999]
alphas = [0.05, 0.1, 0.2]
discretizations = [5, 10, 15]

best_performance = -float('inf')
best_params = None

results = []

for gamma in gammas:
    for eps_decay in epsilon_decays:
        for alpha in alphas:
            for disc in discretizations:
                print(f"Testing: gamma={gamma}, eps_decay={eps_decay}, alpha={alpha}, disc={disc}")                
                performance = train_and_evaluate(gamma, eps_decay, alpha, disc)
                
                results.append({'gamma': gamma,'epsilon_decay': eps_decay,'alpha': alpha,'discretization': disc,'performance': performance})
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = {'gamma': gamma,'epsilon_decay': eps_decay,'alpha': alpha,'discretization': disc}
print(f"Best parameters: {best_params}")
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)