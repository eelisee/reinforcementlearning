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
    num_seeds = len(config['seeds'])
    print(f"Starting minibatch REINFORCE training for {num_seeds} seeds...")

    for seed_idx, seed in enumerate(config['seeds']):
    
        env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        returns = []
        timesteps_list = []
        total_timesteps = 0
        episode_count = 0
        last_eval_timestep = 0
        eval_freq = config.get('eval_freq', 1000)

        while total_timesteps < config['timesteps']:
            batch_log_probs = []
            batch_returns = []

            # Minibatch Episoden sammeln
            while total_timesteps < config['timesteps']:
                batch_log_probs = []
                batch_returns = []
                batch_episode_returns = []  # für Episoden-Returns im Batch

                # Minibatch Episoden sammeln
                for _ in range(config['batch_size']):
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
                            done = True

                    # Berechne Return Gt für die Episode
                    G = 0
                    returns_episode = []
                    for r in rewards[::-1]:
                        G = r + config['gamma'] * G
                        returns_episode.insert(0, G)

                    batch_log_probs.extend(log_probs)
                    batch_returns.extend(returns_episode)
                    batch_episode_returns.append(sum(rewards))  # einzelne Episode Return speichern

                # Policy-Gradienten-Update mit normalisierten Returns
                returns_tensor = torch.tensor(batch_returns, dtype=torch.float32)
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

                loss = 0
                for log_prob, G in zip(batch_log_probs, returns_tensor):
                    loss -= log_prob * G

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Jetzt einen Wert pro Batch speichern
                avg_batch_return = np.mean(batch_episode_returns)
                returns.append(avg_batch_return)
                timesteps_list.append(total_timesteps)

                if total_timesteps - last_eval_timestep >= eval_freq:
                    last_eval_timestep = total_timesteps
                    print(f"  Seed {seed}: Eval at timestep {total_timesteps} - Avg Return: {avg_batch_return:.2f}")
                    save_reward_curve_as_csv(config['env'], "minibatch", seed, timesteps_list, returns)


        all_returns.append(returns)
        save_reward_curve_as_csv(config['env'], "minibatch", seed, timesteps_list, returns)

    save_results(config['env'], 'minibatch', all_returns)
    print("Results saved.")