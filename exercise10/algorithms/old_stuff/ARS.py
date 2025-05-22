import gymnasium as gym
import numpy as np
import torch
from torch import nn

class ARSPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ARSPolicy, self).__init__()
        # Define policy network architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class ARS:
    def __init__(self, env_name, policy_class, population_size=100, noise_sigma=0.1, learning_rate=0.01):
        self.env = gym.make(env_name)
        self.policy_class = policy_class
        self.population_size = population_size
        self.noise_sigma = noise_sigma
        self.learning_rate = learning_rate
        
        # Initialize the policy network
        self.policy = policy_class(self.env.observation_space.shape[0], self.env.action_space.n)
        
    def sample_action(self, state, noise=None):
        with torch.no_grad():
            action_probs = self.policy(torch.FloatTensor(state))
            if noise is not None:
                action_probs += noise
            return action_probs.argmax().item()
    
    def evaluate_policy(self, policy_params):
        # Set policy parameters for evaluation
        self.policy.load_state_dict(policy_params)
        
        total_reward = 0.0
        state, _ = self.env.reset()
        
        while True:
            action = self.sample_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
            state = next_state
        
        return total_reward
    
    def train(self, num_episodes=100, max_steps_per_episode=1000):
        for episode in range(num_episodes):
            # Generate offspring policies by adding random noise
            parent_params = self.policy.state_dict()
            offspring_params = []
            noises = []
            
            for _ in range(self.population_size):
                current_noise = {}
                for param_name in parent_params:
                    current_noise[param_name] = torch.randn_like(parent_params[param_name]) * self.noise_sigma
                child_params = {k: v.clone() + current_noise[k] for k, v in parent_params.items()}
                offspring_params.append(child_params)
                noises.append(current_noise)

            
            # Evaluate all offspring policies
            # rewards = []
            # for params in offspring_params:
            #     self.policy.load_state_dict(params)
            #     episode_reward = 0.0
            #     state, _ = self.env.reset()
                
            #     for _ in range(max_steps_per_episode):
            #         action = self.sample_action(state)
            #         next_state, reward, terminated, truncated, info = self.env.step(action)
            #         episode_reward += reward
                    
            #         if terminated or truncated:
            #             break
            #         state = next_state
                
            #     rewards.append(episode_reward)
            rewards = [self.evaluate_policy(params) for params in offspring_params]

            
            # Compute fitness scores (rewards) and select top performers
            #rewards_np = np.array(rewards)
            #fitness_scores = rewards_np
            
            # Compute the rank-based rewards for gradient estimation
            #ranks = np.argsort(fitness_scores)[::-1]
            
            ranks = np.argsort(rewards)[::-1]  # Highest reward first
            num_top = int(self.population_size * 0.2)  # Take top 20% performers
            top_indices = ranks[:num_top]


            # Calculate the average of the top K policies' noise directions
            #num_top = int(self.population_size * 0.2)  # Take top 20% performers
            #selected_noises = [noises[i] for i in ranks[:num_top]]
            
            #avg_noise = {}
            #for param_name in noises:
            #    avg_noise[param_name] = torch.mean(torch.stack([noise[param_name] for noise in selected_noises]), dim=0)
            
            summed_noise = {k: torch.zeros_like(parent_params[k]) for k in parent_params}
            for idx in top_indices:
                current_noise = noises[idx]
                for param_name in current_noise:
                    summed_noise[param_name] += current_noise[param_name]

            for param_name in summed_noise:
                summed_noise[param_name] /= num_top
            new_parent_params = {k: parent_params[k] + (summed_noise[k] * self.learning_rate) for k in parent_params}
            self.policy.load_state_dict(new_parent_params)


            # Update the parent policy parameters using the average noise direction
            #with torch.no_grad():
            #    for param_name in parent_params:
            #        self.policy.state_dict()[param_name] += self.learning_rate * avg_noise[param_name]
            
            # Print episode statistics
            print(f"Episode {episode + 1}, Average Reward: {np.mean(rewards)}")

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    ars_agent = ARS(env_name, ARSPolicy)
    ars_agent.train(num_episodes=500)
