# eval/plot_metrics.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_metrics(df, output_dir):

    envs = df["env"].unique()
    for env in envs:
        plt.figure(figsize=(12, 8))
        env_df = df[df["env"] == env]

        for (algo, seed), run_df in env_df.groupby(["algorithm", "seed"]):
            print(f"Plotting {env} - {algo} - seed {seed}, reward range: {run_df['rewards'].min()} to {run_df['rewards'].max()}")
            plt.plot(run_df["timesteps"], run_df["rewards"], label=f"{algo} (seed {seed})", alpha=0.7)

        plt.title(f"Reward curves for {env}")
        plt.xlabel("Timesteps")
        plt.ylabel("Rewards")
        plt.legend()

        # Special ylim for Pendulum:
        if env == "Pendulum-v1":
            plt.ylim(df["rewards"].min()-10, df["rewards"].max() + 10)
            plt.xlim(0, run_df["timesteps"].max())
        else:
            plt.autoscale()

        filename = os.path.join(output_dir, f"{env}_reward_plot.png")
        print(f"Saving plot to {filename}")
        plt.savefig(filename)
        plt.close()