import torch
import torch.nn as nn

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, max_action=1.0):
        super(TD3Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
        
    def forward(self, state):
        return self.net(state) * self.max_action

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(TD3Critic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2


class TD3Agent:
    def __init__(self, state_dim, action_dim, hidden_size=256, learning_rate=3e-4, gamma=0.99,
                 tau=0.005, noise_std=0.1, noise_clip=0.5, policy_update_freq=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize networks
        self.actor = TD3Actor(state_dim, action_dim, hidden_size)
        self.critic = TD3Critic(state_dim, action_dim, hidden_size)
        self.target_actor = TD3Actor(state_dim, action_dim, hidden_size)
        self.target_critic = TD3Critic(state_dim, action_dim, hidden_size)
        
        # Copy actor and critic parameters to target networks
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic.parameters():
            param.requires_grad = False
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std  # Standard deviation of the exploration noise
        self.noise_clip = noise_clip  # Clip the noise to prevent too large deviations
        self.policy_update_freq = policy_update_freq  # Frequency of policy updates
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=int(1e6), batch_size=256)
        
        # Counter for delaying policy updates
        self.update_counter = 0

    def sample_action(self, state):
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state).to(device))
            noise = torch.randn_like(action) * self.noise_std
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            action += noise
            action = torch.clamp(action, -1.0, 1.0)  # Ensure actions are within the valid range
        return action.cpu().numpy()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Compute target Q values
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            noise = torch.randn_like(next_actions) * self.noise_std
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions += noise
            next_actions = torch.clamp(next_actions, -1.0, 1.0)
            
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2)
            
            # Compute target Q values
            target_q = rewards + (self.gamma * (1 - dones) * min_target_q)
        
        # Compute current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Optimize the critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Soft update target critic networks
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Delayed policy updates
        self.update_counter += 1
        if self.update_counter % self.policy_update_freq == 0:
            # Compute actor loss
            actions_pred = self.actor(states)
            q_values = self.critic(states, actions_pred)[0]  # Use one of the critics for policy gradient
            actor_loss = -q_values.mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target actor network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
