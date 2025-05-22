# exercise10/environments.py

import os
import torch
import numpy as np
from omegaconf import OmegaConf

from algorithms.common.utils.common_utils import fix_random_seed
from algorithms.common.utils.logger import Logger

# Importiere hier deine Agents
from algorithms.reinforce.agent import ReinforceAgent
from algorithms.a2c.agent import A2CAgent
# weitere folgen...

def run_experiment(agent_type: str, env_name: str = "CartPole-v1", num_seeds: int = 5):
    for seed in range(num_seeds):
        print(f"\n=== Running {agent_type} on {env_name} with seed {seed} ===")
        fix_random_seed(seed)

        # Konfigurationen
        experiment_info = OmegaConf.create({
            "env_name": env_name,
            "env": {
                "name": env_name,
                "is_discrete": True,  # oder False, je nach env
                "state_dim": None,
                "action_dim": None,
            },
            "num_workers": 1,
            "worker_device": "cpu",
            "log_wandb": False,
            "is_discrete": True,
            "max_update_steps": 200,
            "test_interval": 50,
        })

        hyper_params = OmegaConf.create({
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "actor_gradient_clip": 0.5,
            "critic_gradient_clip": 0.5,
            "gamma": 0.99,
            "alpha": 0.01,
        })

        model_cfg = OmegaConf.create({
            "actor": {
                "name": "mlp_policy",
                "params": {
                    "model_cfg": {
                        "hidden_dims": [64, 64],
                        "activation": "relu",
                        "state_dim": None,  # wird im Agent gesetzt
                        "action_dim": None,
                    }
                }
            },
            "critic": {
                "name": "mlp_value",
                "params": {
                    "model_cfg": {
                        "hidden_dims": [64, 64],
                        "activation": "relu",
                        "state_dim": None,
                        "action_dim": None,
                    }
                }
            }
        })

        # Auswahl des Agent-Typs
        if agent_type.lower() == "reinforce":
            agent = ReinforceAgent(experiment_info, hyper_params, model_cfg)
        elif agent_type.lower() == "a2c":
            agent = A2CAgent(experiment_info, hyper_params, model_cfg)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent.train()

if __name__ == "__main__":
    # Beispiel: Reinforce auf CartPole mit 5 Seeds
    run_experiment(agent_type="reinforce", env_name="CartPole-v1", num_seeds=5)