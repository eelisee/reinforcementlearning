# models/reinforce/reinforce.py

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.logger import save_results, save_reward_curve_as_csv

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def train(config):
    env = gym.make(config['env'])
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=config['lr'])

    all_returns = []
    for seed_idx, seed in enumerate(config['seeds']):
        print(f"\nStarting training for seed {seed} ({seed_idx+1}/{len(config['seeds'])})")
        env.reset(seed=seed)
        torch.manual_seed(seed)

        total_timesteps = 0
        returns = []
        timesteps_list = []
        last_eval_timestep = 0

        while total_timesteps < config['timesteps']:
            log_probs = []
            rewards = []
            state, _ = env.reset()
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                probs = policy(state_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                state, reward, done, _, _ = env.step(action.item())
                rewards.append(reward)

                total_timesteps += 1
                if total_timesteps >= config['timesteps']:
                    done = True  # Training stoppen wenn Limit erreicht

            episode_return = sum(rewards)
            returns.append(episode_return)
            timesteps_list.append(total_timesteps)

            # Berechnung Loss
            G = 0
            loss = 0
            for log_prob, r in zip(log_probs[::-1], rewards[::-1]):
                G = r + config['gamma'] * G
                loss -= log_prob * G

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            eval_freq = config.get('eval_freq', 1000)
            
            if total_timesteps - last_eval_timestep >= eval_freq:
                last_eval_timestep = total_timesteps

    
                # Zwischenergebnisse speichern, z.B. CSV mit bisherigen timesteps & returns
                save_reward_curve_as_csv(config['env'], "reinforce", seed, timesteps_list, returns)

        all_returns.append(returns)

        # Am Ende CSV speichern mit Reward vs Timesteps
        save_reward_curve_as_csv(config['env'], "reinforce", seed, timesteps_list, returns)

    save_results(config['env'], 'reinforce', all_returns)
    print("Results saved.")