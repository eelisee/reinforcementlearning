# utils/env_utils.py

import gymnasium as gym
from utils.seed_utils import set_global_seed

def make_env(env_id, seed=None):
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
        set_global_seed(seed)
    return env
