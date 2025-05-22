import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        probs = torch.softmax(self.fc2(x), dim=-1)
        return probs

class MiniBatchREINFORCE:
    def __init__(self, env_name, gamma=0.99, learning_rate=0.01, batch_size=5):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.policy = PolicyNetwork(state_dim, action_dim)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Tracking variables for a batch of episodes
        self.batch_rewards = []
        self.batch_log_probs = []

    def act(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def update_policy(self):
        # Compute returns for each episode in the batch
        all_returns = []
        for rewards in self.batch_rewards:
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            all_returns.append(returns)
        
        # Flatten returns and log_probs
        flat_returns = [ret for eps_ret in all_returns for ret in eps_ret]
        flat_log_probs = [lp for eps_lp in self.batch_log_probs for lp in eps_lp]
        
        # Convert to tensors
        returns_tensor = torch.FloatTensor(flat_returns)
        log_probs_tensor = torch.stack(flat_log_probs)
        
        # Normalize returns if batch size is greater than 1
        if len(flat_returns) > 1:
            returns_mean = returns_tensor.mean()
            returns_std = returns_tensor.std() + 1e-7
            returns_tensor = (returns_tensor - returns_mean) / returns_std
        
        # Compute policy loss
        policy_loss = (-log_probs_tensor * returns_tensor).sum()
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Reset batch tracking variables
        self.batch_rewards = []
        self.batch_log_probs = []

    def train_episode(self):
        state, _ = self.env.reset()
        episode_rewards = []
        episode_log_probs = []
        episode_return = 0
        
        for _ in range(1000):  # Maximum steps per episode
            action, log_prob = self.act(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            episode_return += reward
            
            if terminated or truncated:
                break
                
            state = next_state
        
        # Add the completed episode to the batch
        self.batch_rewards.append(episode_rewards)
        self.batch_log_probs.append(episode_log_probs)
        
        # Update policy after a full batch is collected
        if len(self.batch_rewards) == self.batch_size:
            self.update_policy()
        
        return episode_return

# Example usage:
if __name__ == "__main__":
    env_name = 'CartPole-v1'
    reinforce_agent = MiniBatchREINFORCE(env_name)
    
    total_episodes = 500
    batch_size = 5
    episodes_per_batch = batch_size
    returns = []

    for episode in range(total_episodes):
        episode_return = reinforce_agent.train_episode()
        returns.append([episode + 1, episode_return])
        
        # Print progress every few batches
        if (episode + 1) % episodes_per_batch == 0:
            print(f"Episode {episode + 1}, Batch Update: Return = {episode_return}")
    
    # Optional: Save the trained policy network
    torch.save(reinforce_agent.policy.state_dict(), 'exercise10/algorithms/policies/minibatch_reinforce_policy.pth')

    # Save returns to CSV
    csv_path = 'exercise10/algorithms/returns/minibatch_reinforce_returns.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'return'])
        writer.writerows(returns)
