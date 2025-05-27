# utils/config_loader.py
import os
import yaml

import os
import yaml

# Algorithmen für diskrete bzw. kontinuierliche Action-Spaces
DISCRETE_ALGOS = {"reinforce", "minibatch", "ars", "a2c", "ppo", "trpo"}
CONTINUOUS_ALGOS = {"ddpg", "sac", "td3", "tqc"}

# Hilfsfunktion, um zu bestimmen, ob eine Umgebung diskret ist
# (hier hardcoded, kann erweitert oder aus Gym abgefragt werden)
DISCRETE_ENVS = {"CartPole-v1", "Acrobot-v1", "Taxi-v3"}#, "MountainCar-v0"}
CONTINUOUS_ENVS = {"Pendulum-v1", "MountainCarContinuous-v0"}#, "LunarLanderContinuous-v2"}

def load_all_configs():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Pfad zu utils/
    base_path = os.path.join(base_dir, "..", "configs", "classic_control")
    base_path = os.path.normpath(base_path)

    configs = []
    for filename in os.listdir(base_path):
        if filename.endswith(".yaml"):
            full_path = os.path.join(base_path, filename)
            with open(full_path, "r") as f:
                data = yaml.safe_load(f)

                env = data["env"]

                # Entscheide, ob env diskret oder kontinuierlich
                if env in DISCRETE_ENVS:
                    allowed_algos = DISCRETE_ALGOS
                elif env in CONTINUOUS_ENVS:
                    allowed_algos = CONTINUOUS_ALGOS
                else:
                    # Falls unbekannt, nehme alle Algorithmen aus Config, warnen optional
                    print(f"[Warning] Environment '{env}' unbekannt, keine Filterung angewandt.")
                    allowed_algos = set(data["algorithms"])

                for algo in data["algorithms"]:
                    if algo not in allowed_algos:
                        # Algorithmus passt nicht zum Action-Space, skip
                        continue

                    config = {
                        "env": env,
                        "algorithm": algo,
                        "lr": data.get("lr", 0.001),
                        "gamma": data.get("gamma", 0.99),
                        "episodes": data.get("episodes", 100),
                        "timesteps": data.get("timesteps", 500),
                        "batch_size": data.get("batch_size", 64),
                        "eval_episodes": data.get("eval_episodes", 5),
                        "eval_freq": data.get("eval_freq", 1000),  # Optional, falls nicht angegeben
                        "seeds": data.get("seeds", [0 , 1]),#, 2, 3, 4]),
                    }
                    configs.append(config)
    return configs



# def load_all_configs():
#     base_dir = os.path.dirname(os.path.abspath(__file__))  # Pfad zu utils/
#     base_path = os.path.join(base_dir, "..", "configs", "classic_control")
#     base_path = os.path.normpath(base_path)

#     configs = []
#     for filename in os.listdir(base_path):
#         if filename.endswith(".yaml"):
#             full_path = os.path.join(base_path, filename)
#             with open(full_path, "r") as f:
#                 data = yaml.safe_load(f)
#                 for algo in data["algorithms"]:
#                     config = {
#                         "env": data["env"],
#                         "algorithm": algo,
#                         "lr": data.get("lr", 0.001),
#                         "gamma": data.get("gamma", 0.99),
#                         "episodes": data.get("episodes", 100),
#                         "timesteps": data.get("timesteps", 500),
#                         "batch_size": data.get("batch_size", 64),
#                         "eval_episodes": data.get("eval_episodes", 5),
#                         "seeds": data.get("seeds", [0]), #, 1, 2, 3, 4]),
#                     }
#                     configs.append(config)
#     return configs