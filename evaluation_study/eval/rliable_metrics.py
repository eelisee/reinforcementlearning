import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aggregate_metrics import aggregate_reward_curves

sns.set(style="whitegrid")

TARGET_TIMESTEP = 50000
N_BOOTSTRAP_SAMPLES = 1000
ALPHA = 0.95


def interpolate_reward(df, target_step):
    steps = df['timesteps'].values
    rewards = df['rewards'].values
    if len(steps) == 0:
        return np.nan
    if target_step in steps:
        return df[df['timesteps'] == target_step]['rewards'].values[0]
    elif target_step < steps[0] or target_step > steps[-1]:
        return np.nan
    else:
        return np.interp(target_step, steps, rewards)


def extract_final_rewards_per_env(df_all, timestep=TARGET_TIMESTEP):
    per_env_data = {}

    grouped = df_all.groupby(["env", "algorithm", "seed"])

    for (env, algo, seed), group in grouped:
        reward = interpolate_reward(group, timestep)
        if np.isnan(reward):
            print(f"SKIPPED: {env} - {algo} - seed {seed} (timestep range: {group['timesteps'].min()} to {group['timesteps'].max()})")
            continue

        if env not in per_env_data:
            per_env_data[env] = {}
        if algo not in per_env_data[env]:
            per_env_data[env][algo] = []

        per_env_data[env][algo].append(reward)

    return per_env_data


def compute_iqm(scores):
    sorted_scores = np.sort(scores)
    q25, q75 = np.percentile(sorted_scores, [25, 75])
    iqm_scores = sorted_scores[(sorted_scores >= q25) & (sorted_scores <= q75)]
    return np.mean(iqm_scores) if len(iqm_scores) > 0 else np.nan


def bootstrap_interval_estimates(data, metric="iqm", num_bootstrap_samples=1000, confidence_level=0.95):
    rng = np.random.default_rng()
    estimates = {}

    for algo, scores in data.items():
        if len(scores) < 2:
            continue

        boot_samples = []
        scores = np.array(scores)
        for _ in range(num_bootstrap_samples):
            sample = rng.choice(scores, size=len(scores), replace=True)
            if metric == "iqm":
                boot_val = compute_iqm(sample)
            else:
                raise ValueError("Unsupported metric")
            boot_samples.append(boot_val)

        boot_samples = np.array(boot_samples)
        lower = np.percentile(boot_samples, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(boot_samples, (1 + confidence_level) / 2 * 100)
        median = np.median(boot_samples)
        estimates[algo] = (median, lower, upper)

    return estimates


def plot_interval_estimates(iqm_dict, ylabel="IQM", xlabel="Algorithm", title=None):
    if len(iqm_dict) < 2:
        return None, None

    algos = list(iqm_dict.keys())
    medians = [iqm_dict[a][0] for a in algos]
    lowers = [iqm_dict[a][1] for a in algos]
    uppers = [iqm_dict[a][2] for a in algos]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(algos, medians,
                yerr=[np.subtract(medians, lowers), np.subtract(uppers, medians)],
                fmt='o', capsize=5, elinewidth=2, markeredgewidth=2)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title if title else f"{ylabel} with Confidence Intervals")

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    plt.tight_layout()
    return fig, ax


def plot_performance_profiles(data, title="Performance Profiles"):
    # Filter Algos with at least 1 score
    valid_data = {k: np.array(v) for k, v in data.items() if len(v) > 0}
    if len(valid_data) < 2:
        return None, None

    # Global normalization
    all_scores = np.concatenate(list(valid_data.values()))
    max_score = np.max(all_scores)
    epsilon = 1e-8
    normalized = {k: v / (max_score + epsilon) for k, v in valid_data.items()}

    taus = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, scores in normalized.items():
        profile = [np.mean(scores >= tau) for tau in taus]
        ax.plot(taus, profile, label=algo)

    ax.set_xlabel("Normalized Score Threshold (τ)")
    ax.set_ylabel("Fraction of Runs ≥ τ")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_all_envs_ci(per_env_rewards, output_path):
    for env, rewards in per_env_rewards.items():
        ci_iqm = bootstrap_interval_estimates(rewards, metric="iqm",
                                              num_bootstrap_samples=N_BOOTSTRAP_SAMPLES,
                                              confidence_level=ALPHA)
        if not ci_iqm:
            continue
        fig, ax = plot_interval_estimates(ci_iqm, ylabel="IQM", xlabel="Algorithm",
                                          title=f"95% CI of IQM - {env}")
        if fig:
            fig.savefig(os.path.join(output_path, f"{env}_ci_iqm.png"))
            plt.close(fig)


def plot_all_envs_performance_profiles(per_env_rewards, output_path):
    for env, rewards in per_env_rewards.items():
        fig, ax = plot_performance_profiles(rewards, title=f"Performance Profiles - {env}")
        if fig:
            fig.savefig(os.path.join(output_path, f"{env}_performance_profile.png"))
            plt.close(fig)


def main():
    base_path = "evaluation_study/data"
    output_path = "evaluation_study/plots"
    os.makedirs(output_path, exist_ok=True)

    print("Lade aggregierte Rewards...")
    df_all = aggregate_reward_curves(base_path)

    print("Extrahiere Rewards pro Environment...")
    per_env_rewards = extract_final_rewards_per_env(df_all)

    print("Plotte CI-IQM Plots...")
    plot_all_envs_ci(per_env_rewards, output_path)

    print("Plotte Performance Profile Plots...")
    plot_all_envs_performance_profiles(per_env_rewards, output_path)

    print("Fertig! Plots gespeichert in:", output_path)


if __name__ == "__main__":
    main()