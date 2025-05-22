import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class REINFORCE:
    def __init__(self, env_name, gamma=0.99, learning_rate=0.01):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Initialize policy network
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.policy = PolicyNetwork(state_dim, action_dim)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Tracking variables
        self.rewards = []
        self.log_probs = []

    def act(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.log_probs.append(log_prob)
        return action.item()

    def update_policy(self):
        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns_tensor = torch.FloatTensor(returns)
        # Normalize returns for stability
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-7)
        
        # Compute loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns_tensor):
            policy_loss.append(-log_prob * R)
            
        total_loss = torch.stack(policy_loss).sum()
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Reset tracking variables
        self.rewards = []
        self.log_probs = []

    def train_episode(self):
        state, _ = self.env.reset()
        episode_return = 0
        
        for _ in range(1000):  # Maximum steps per episode
            action = self.act(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            self.rewards.append(reward)
            episode_return += reward
            
            if terminated or truncated:
                break
                
            state = next_state
        
        # Update policy after the episode
        self.update_policy()
        
        return episode_return

# Example usage:
if __name__ == "__main__":
    env_name = 'CartPole-v1'
    reinforce_agent = REINFORCE(env_name)
    returns = []

    for episode in range(500):
        episode_return = reinforce_agent.train_episode()
        returns.append((episode + 1, episode_return))
        print(f"Episode {episode + 1}: Return = {episode_return}")

    # Optional: Save the trained policy network
    torch.save(reinforce_agent.policy.state_dict(), 'exercise10/algorithms/policies/reinforce_policy.pth')

    # Save returns to CSV
    csv_path = 'exercise10/algorithms/returns/reinforce_returns.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'return'])
        writer.writerows(returns)
