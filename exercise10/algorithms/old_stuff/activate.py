import gymnasium as gym
from collections import deque 

from PPO import PPOPolicy
from DDPG import DDPGAgent
from TRPO import TRPO_Agent
from TQC import TQC_Agent
#from TD3 import TD3Agent


# Train TQC Agent
# agent = TQC_Agent(env_name='Pendulum-v1')
# replay_buffer = deque(maxlen=100000)

# for episode in range(100):
#     state = agent.env.reset()
#     total_reward = 0
#     for step in range(200):
#         action = agent.act(state)
#         next_state, reward, done, _ = agent.env.step(action)
#         replay_buffer.append((state, action, reward, next_state, done))
        
#         if len(replay_buffer) > 1000:
#             agent.train(replay_buffer, batch_size=128)
        
#         total_reward += reward
#         state = next_state
        
#     print(f'Episode {episode}, Total Reward: {total_reward}')

# Train TRPO Agent
agent = TRPO_Agent(env_name='Pendulum-v1')

for episode in range(100):
    states, actions, rewards, dones, next_states = agent.collect_trajectories()
    agent.train()
    
    total_reward = sum(rewards)
    print(f'Episode {episode}, Total Reward: {total_reward}')


# env = gym.make('CartPole-v1')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]

# #gent = DDPGAgent(state_dim, action_dim)

# agent = TRPO_Agent(state_dim, action_dim)

# for episode in range(1000):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     while not done:
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         agent.remember(state, action, reward, next_state, done)
#         total_reward += reward
#         state = next_state
        
#         if len(agent.memory) > 1000:
#             agent.train()
            
#     print(f"Episode {episode}, Total Reward: {total_reward}")
    
# agent.save_model("TRPO_cartpole")


# env = gym.make('')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

# policy = PPOPolicy(state_dim, action_dim)

# for episode in range(1000):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     experiences = []
    
#     while not done:
#         action, log_prob = policy.get_action(state)
#         next_state, reward, done, _ = env.step(action)
#         experiences.append({
#             'state': state,
#             'action': action,
#             'reward': reward,
#             'next_state': next_state,
#             'log_prob': log_prob
#         })
#         total_reward += reward
#         state = next_state
        
#     policy.train(experiences)
#     print(f"Episode {episode}, Total Reward: {total_reward}")
    
# policy.save_model("ppo_cartpole")

# ##################

# # For SAC


# # For TD3
# agent = TD3Agent(state_dim, action_dim)



# for episode in range(num_episodes):
#     state = env.reset()
#     total_reward = 0
    
#     while True:
#         # Sample action
#         action = agent.sample_action(state)
        
#         # Execute action and observe next state and reward
#         next_state, reward, done, _ = env.step(action)
        
#         # Store experience in replay buffer
#         agent.replay_buffer.add(state, action, reward, next_state, done)
        
#         # Train agent
#         if len(agent.replay_buffer) >= agent.batch_size:
#             agent.train()
            
#         total_reward += reward
#         state = next_state
        
#         if done:
#             break
    
#     print(f"Episode {episode}: Reward {total_reward}")


# # Save model
# agent.save_model("model.pth")