import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import A2C, PPO


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


class REINFORCE:
    def __init__(self, env, gamma=0.99, lr=0.01):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def update(self):
        R = 0
        returns = []
        
        for r in reversed(self.episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        policy_loss = -torch.stack(self.episode_log_probs) * torch.FloatTensor(returns)
        loss = policy_loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
    
    def train_episode(self):
        state, _ = self.env.reset()
        done = False
        
        while not done:
            action, log_prob = self.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            self.episode_rewards.append(reward)
            self.episode_log_probs.append(log_prob)
            
            state = next_state
        
        self.update()


class MiniBatchREINFORCE(REINFORCE):
    def __init__(self, env, gamma=0.99, lr=0.01, batch_size=5):
        super().__init__(env, gamma, lr)
        self.batch_size = batch_size
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
    
    def train_batch(self, num_episodes):
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action, log_prob = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.episode_rewards.append(reward)
                self.episode_log_probs.append(log_prob)
                
                state = next_state
            
            if len(self.episode_rewards) >= self.batch_size:
                self.update()


def evaluate_algorithm(algorithm_class, env_name, num_seeds=5):
    env = gym.make(env_name)
    returns = []
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        agent = algorithm_class(env)
        
        total_return = 0
        for _ in range(100):  # Number of episodes per run
            state, _ = env.reset()
            episode_return = 0
            done = False
            
            while not done:
                action, _ = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward
                
                state = next_state
            
            total_return += episode_return
        
        returns.append(total_return / 100)  # Average return per episode
    
    return returns



import rliable as rl

def evaluate_algorithm(algorithm_class, env_name, num_seeds=5):
    env = gym.make(env_name)
    returns = []

    for seed in range(num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = algorithm_class(env)

        total_return = 0
        for _ in range(100):  # Number of episodes per run
            state, _ = env.reset()
            episode_return = 0
            done = False

            while not done:
                action, _ = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward

                state = next_state

            total_return += episode_return

        returns.append(total_return / 100)  # Average return per episode

    # Compute reliable metrics
    metrics = rl.metrics.compute(returns)
    print(metrics)

    return returns
