# utils/logger.py

import os
import json
import numpy as np
import pandas as pd

def save_results(env, algo, returns):
    os.makedirs(f"evaluation_study/data/classic_control/{env}", exist_ok=True)
    out_path = f"evaluation_study/data/classic_control/{env}/{algo}.json"
    with open(out_path, 'w') as f:
        json.dump(returns, f)

def save_eval_as_csv(log_path):
    eval_file = os.path.join(log_path, "evaluations.npz")
    if os.path.exists(eval_file):
        data = np.load(eval_file)
        timesteps = data['timesteps']
        results = data['results']  # Array der Rewards

        # Manche 'results' sind 2D (Mehrere Episoden pro Zeitpunkt), Mittelwert nehmen
        if len(results.shape) > 1:
            mean_rewards = np.mean(results, axis=1)
        else:
            mean_rewards = results

        df = pd.DataFrame({
            "timesteps": timesteps,
            "mean_reward": mean_rewards
        })

        csv_path = os.path.join(log_path, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
    else:
        print(f"Evaluations file not found at {eval_file}")

def save_reward_curve_as_csv(env_name, algo_name, seed, timesteps, rewards):
    """
    Speichert Rewards und Timesteps als CSV-Datei
    Pfad: evaluation_study/data/classic_control/{env_name}_{algo_name}_{seed}/
    """
    folder = f"evaluation_study/data/classic_control/{env_name}_{algo_name}_{seed}"
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame({"timesteps": timesteps, "rewards": rewards})
    csv_path = os.path.join(folder, "reward_curve.csv")
    df.to_csv(csv_path, index=False)