import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class TQC_Agent:
    def __init__(self, env_name, gamma=0.99, tau=0.005, alpha=0.2,
                 hidden_size=256, actor_lr=3e-4, critic_lr=3e-4):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Initialize networks
        self.actor = PolicyNetwork(self.state_dim, self.action_dim, hidden_size)
        self.critic1 = QNetwork(self.state_dim, self.action_dim, hidden_size)
        self.critic2 = QNetwork(self.state_dim, self.action_dim, hidden_size)
        
        # Target networks
        self.target_actor = PolicyNetwork(self.state_dim, self.action_dim, hidden_size)
        self.target_critic1 = QNetwork(self.state_dim, self.action_dim, hidden_size)
        self.target_critic2 = QNetwork(self.state_dim, self.action_dim, hidden_size)
        
        # Copy weights from original networks
        for target_param, param in zip(self.target_actor.parameters(), 
                                      self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic1.parameters(),
                                      self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic2.parameters(),
                                      self.critic2.parameters()):
            target_param.data.copy_(param.data)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
    
    def act(self, state):
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state))
        return action.numpy()
    
    def train(self, replay_buffer, batch_size=100):
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        
        # Compute target Q values using target networks
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * self.actor(next_states).log_prob(actions).exp()
        
        # Compute current Q estimates
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Compute critic losses
        loss_critic1 = F.mse_loss(current_q1, rewards + (1 - dones) * self.gamma * target_q)
        loss_critic2 = F.mse_loss(current_q2, rewards + (1 - dones) * self.gamma * target_q)
        
        # Optimize critics
        self.critic1_optimizer.zero_grad()
        loss_critic1.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        loss_critic2.backward()
        self.critic2_optimizer.step()
        
        # Delayed policy update
        if self.train_step % 2 == 0:
            # Compute actor loss
            actions_pi = self.actor(states)
            q1 = self.critic1(states, actions_pi)
            q2 = self.critic2(states, actions_pi)
            min_q = torch.min(q1, q2)
            policy_loss = -min_q.mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.target_actor.parameters(), 
                                      self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic1.parameters(),
                                      self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(),
                                      self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.train_step += 1

    def save_model(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, filename)
    
    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
