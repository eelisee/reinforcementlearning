import torch
import torch.nn as nn
import torch.optim as optim, torch.nn.functional as F
import torch.distributions
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class TRPO_Agent:
    def __init__(self, env_name, gamma=0.99, lambda_=0.98,
                 delta=0.01, hidden_size=64):
        self.env = gym.make(env_name)
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta = delta
        
        # Network and optimizer
        self.policy = Policy(self.env.observation_space.shape[0],
                            self.env.action_space.shape[0], hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # Other parameters
        self.K = 10  # Number of steps per update
        
    def act(self, state):
        with torch.no_grad():
            action = self.policy(torch.FloatTensor(state))
        return action.numpy()
    
    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i]
                advantage = delta
            else:
                delta = rewards[i] + self.gamma * values[i+1] - values[i]
                advantage = delta + self.gamma * self.lambda_ * advantage
            advantages.append(advantage)
        return torch.FloatTensor(advantages[::-1])
    
    def fisher_vector_product(self, params, v):
        kl = self.compute_kl()
        kl_grad = torch.autograd.grad(kl, params, create_graph=True)
        kl_grad_v = sum(g * vg for g, vg in zip(kl_grad, v))
        return torch.autograd.grad(kl_grad_v, params)[0] + 0.1*v
    
    def compute_kl(self):
        with torch.no_grad():
            old_pi = self.policy(torch.FloatTensor(self.states)).data
        new_pi = self.policy(torch.FloatTensor(self.states))
        kl = F.kl_div(new_pi.log_prob(old_pi).exp().log(), 
                      torch.distributions.Normal(old_pi, 1e-8),
                      reduction='batchmean')
        return kl
    
    def train(self):
        # Collect trajectories
        states, actions, rewards, dones, next_states = self.collect_trajectories()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        dones = torch.FloatTensor(np.array(dones))
        next_states = torch.FloatTensor(np.array(next_states))
        
        # Compute values
        with torch.no_grad():
            values = []
            current_value = 0
            for i in reversed(range(len(rewards))):
                if dones[i]:
                    current_value = rewards[i]
                else:
                    current_value = rewards[i] + self.gamma * current_value
                values.append(current_value)
            values = values[::-1]
            target_values = torch.FloatTensor(values[:-1])
        
        # Compute advantages
        with torch.no_grad():
            current_pi = self.policy(states)
            actions_log_prob = current_pi.log_prob(actions).sum(-1)
        advantages = self.compute_advantages(rewards, 
                                           np.append(values[:-1], 0),
                                           values[-1], dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute gradients
        policy_loss = -(actions_log_prob * advantages).mean()
        grads = torch.autograd.grad(policy_loss, self.policy.parameters())
        
        # Natural gradient descent
        fisher = self.fisher_vector_product(self.policy.parameters(), grads[0])
        step = -self.delta * grads[0] / (fisher @ grads[0] + 1e-8).sqrt()
        
        with torch.no_grad():
            new_theta = list(self.policy.parameters())[0] + step
            kl_after = self.compute_kl()
            
            if kl_after < self.delta:
                list(self.policy.parameters())[0].data.copy_(new_theta)
            else:
                pass
        
    def collect_trajectories(self, max_steps=1000):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        state = self.env.reset()
        
        for _ in range(max_steps):
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
        
        return states, actions, rewards, dones, next_states
    
    def save_model(self, filename):
        torch.save({
            'policy': self.policy.state_dict(),
        }, filename)
    
    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy'])
