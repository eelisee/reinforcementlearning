# models/sota/sac_wrapper.py

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from utils.env_utils import make_env
from utils.logger import save_results, save_reward_curve_as_csv
import numpy as np
import os

def train(config):
    all_returns = []
    num_seeds = len(config['seeds'])
    print(f"Starting SAC training for {num_seeds} seeds...")

    for idx, seed in enumerate(config['seeds']):
        env = make_env(config['env'], seed)
        model = SAC('MlpPolicy', env, seed=seed, verbose=0)

        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"evaluation_study/logs/{config['env']}_SAC_{seed}/",
            log_path=f"evaluation_study/logs/{config['env']}_SAC_{seed}/",
            eval_freq=100,
            deterministic=True,
            render=False
        )

        model.learn(total_timesteps=config['timesteps'], callback=eval_callback)

        # eval_callback schreibt in log_path/evaluations.npz
        eval_npz_path = os.path.join(f"evaluation_study/logs/{config['env']}_SAC_{seed}", "evaluations.npz")
        if os.path.exists(eval_npz_path):
            data = np.load(eval_npz_path)
            timesteps = data['timesteps']
            rewards = data['results']
            if rewards.ndim > 1:
                rewards = np.mean(rewards, axis=1)  # Mittelwert falls mehrere Episoden pro eval
            save_reward_curve_as_csv(config['env'], "sac", seed, timesteps, rewards)
        else:
            print(f"[Warning] evaluations.npz nicht gefunden unter {eval_npz_path}")

        # Für Summary speichern wir nur den Mittelwert der letzten Evaluation als Return
        last_mean_return = np.mean(rewards[-config.get('eval_episodes', 10):]) if len(rewards) >= config.get('eval_episodes', 10) else np.mean(rewards)
        all_returns.append(last_mean_return)

    save_results(config['env'], 'sac', all_returns)
    print("Training complete.")