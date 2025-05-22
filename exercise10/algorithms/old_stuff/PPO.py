import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, learning_rate=0.001):
        super(PPOPolicy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Output layers for policy and value function
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state):
        features = self.layers(state)
        policy = torch.nn.Softmax(dim=-1)(self.policy_head(features))
        value = self.value_head(features)
        return policy, value
    
    def get_action(self, state):
        with torch.no_grad():
            policy, _ = self.forward(torch.FloatTensor(state))
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
        return action.item(), policy[action].item()
    
    def compute_loss(self, states, actions, rewards, values, old_log_probs, gamma=0.99, epsilon=0.2):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Calculate advantages
        advantages = rewards - values
        
        # Get current policy and log probabilities
        new_policy, new_values = self.forward(states)
        action_masks = torch.zeros_like(new_policy).scatter_(1, actions.unsqueeze(1), 1)
        new_log_probs = (new_policy.log() * action_masks).sum(dim=1)
        
        # Compute ratio for PPO clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Policy loss
        policy_loss_clipped = torch.min(ratio * advantages, 
                                       torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages).mean()
        
        # Value loss (MSE)
        value_loss = nn.MSELoss()(new_values.squeeze(), rewards)
        
        # Total loss
        total_loss = -policy_loss_clipped + 0.5 * value_loss
        
        return total_loss
    
    def train(self, experiences):
        states = [e['state'] for e in experiences]
        actions = [e['action'] for e in experiences]
        rewards = [e['reward'] for e in experiences]
        next_states = [e['next_state'] for e in experiences]
        old_log_probs = [e['log_prob'] for e in experiences]
        
        # Calculate values (approximated by the same network)
        with torch.no_grad():
            _, values = self.forward(torch.FloatTensor(next_states))
            values = values.squeeze().numpy()
            
        # Compute loss
        loss = self.compute_loss(states, actions, rewards, values, old_log_probs)
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
