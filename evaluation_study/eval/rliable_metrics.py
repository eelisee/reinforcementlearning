# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# import metrics, library
# #from library import StratifiedBootstrap, StratifiedIndependentBootstrap, get_interval_estimates, run_score_deviation, mean_score_deviation, create_performance_profile
# #from metrics import aggregate_mean, aggregate_median, aggregate_optimality_gap, aggregate_iqm, probability_of_improvement
# #from plot_utils import _non_linear_scaling, _decorate_axis, _annotate_and_decorate_axis, plot_performance_profiles, plot_interval_estimates, plot_sample_efficiency_curve, plot_probability_of_improvement

# from aggregate_metrics import aggregate_reward_curves


# def interpolate_rewards(df, timesteps=np.arange(1, 50001)):
#     """
#     Interpoliert Rewards für jeden (env, algorithm, seed)-Triplet auf die gewünschten Timesteps.
#     """
#     records = []
#     grouped = df.groupby(["env", "algorithm", "seed"])

#     for (env, algo, seed), group in grouped:
#         f = interp1d(group["timesteps"], group["rewards"], bounds_error=False, fill_value="extrapolate")
#         rewards_interp = f(timesteps)

#         temp_df = pd.DataFrame({
#             "timesteps": timesteps,
#             "rewards": rewards_interp,
#             "env": env,
#             "algorithm": algo,
#             "seed": seed,
#         })
#         records.append(temp_df)

#     return pd.concat(records, ignore_index=True)


# def prepare_rliable_data(df_interp, final_timestep=50000):
#     """
#     Wandelt den DataFrame in das Format für rliable um:
#     - pro Environment: dict algorithm -> np.array(n_seeds)
#     """
#     results = {}
#     envs = df_interp["env"].unique()
#     for env in envs:
#         env_df = df_interp[df_interp["env"] == env]
#         algorithms = env_df["algorithm"].unique()
#         algo_results = {}

#         for algo in algorithms:
#             # Werte des letzten Zeitschritts für alle Seeds
#             seeds = env_df[env_df["algorithm"] == algo]["seed"].unique()
#             seed_rewards = []
#             for seed in seeds:
#                 rewards = env_df[
#                     (env_df["algorithm"] == algo) &
#                     (env_df["seed"] == seed) &
#                     (env_df["timesteps"] == final_timestep)
#                 ]["rewards"].values
#                 if rewards.size == 1:
#                     seed_rewards.append(rewards[0])
#                 else:
#                     # Falls der exakte Timestep nicht da ist, fallback
#                     # z.B. Mittelwert der nahesten 2 Punkte
#                     subset = env_df[
#                         (env_df["algorithm"] == algo) &
#                         (env_df["seed"] == seed)
#                     ].sort_values("timesteps")
#                     seed_rewards.append(subset["rewards"].iloc[-1])  # fallback letzter Wert

#             algo_results[algo] = np.array(seed_rewards)
#         results[env] = algo_results

#     return results



# # Angepasste Importe – gehen davon aus, dass du ein Package draus machst.


# # # Basispfad zu den Daten
# # BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

# # # Beispielhafte Ladefunktion – hier müsste dein konkreter Ladeprozess stehen
# # def load_scores(file_path):
# #     """Lädt Scores im npy-Format."""
# #     return np.load(file_path)  # shape: (num_runs, num_tasks)

# # # Lade deine Daten
# # def load_all_scores():
# #     return {
# #         'MethodA': load_scores(os.path.join(BASE_PATH, 'method_a.npy')),
# #         'MethodB': load_scores(os.path.join(BASE_PATH, 'method_b.npy')),
# #     }

# # Main Evaluation
# def main():
#     scores = load_all_scores()

#     # Interquartilsmittel
#     iqm_func = lambda x: np.array([metrics.aggregate_iqm(x)])
#     iqm_estimates, iqm_cis = library.get_interval_estimates(
#         scores,
#         func=iqm_func,
#         method='percentile',
#         reps=5000,
#         confidence_interval_size=0.95,
#         task_bootstrap=False
#     )

#     print("Interquartilsmittel mit 95%-Konfidenzintervallen:")
#     for method, val in iqm_estimates.items():
#         ci = iqm_cis[method]
#         print(f"{method}: {val[0]:.3f} (CI: {ci[0][0]:.3f} – {ci[1][0]:.3f})")

#     # Probability of Improvement
#     prob_func = lambda x, y: np.array([metrics.probability_of_improvement(x, y)])
#     prob_estimates, prob_cis = library.get_interval_estimates(
#         {'MethodA_vs_B': [scores['MethodA'], scores['MethodB']]},
#         func=prob_func,
#         method='percentile',
#         reps=5000,
#         confidence_interval_size=0.95,
#     )

#     prob = prob_estimates['MethodA_vs_B'][0]
#     ci = prob_cis['MethodA_vs_B']
#     print(f"\nWahrscheinlichkeit, dass A besser ist als B: {prob:.3f} (CI: {ci[0][0]:.3f} – {ci[1][0]:.3f})")

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import pandas as pd

# Lokale Imports
import aggregate_metrics, metrics, library

# Basispfad – zeigt auf "evaluation_study/data"
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Die Environments (optional, falls du gezielt nur diese auswerten willst)
ENVS = [
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "Taxi-v3",
]

def prepare_score_dict(df, envs):
    """Wandelt aggregierte DataFrame in ein Score-Dict pro Methode."""
    score_dict = {}

    for algo in df["algorithm"].unique():
        algo_scores = []

        for env in envs:
            subset = df[(df["algorithm"] == algo) & (df["env"] == env)]

            if subset.empty:
                continue

            # Nehme den letzten Eintrag aus jeder Reward-Curve pro Seed
            scores = subset.groupby("seed")["rewards"].last().values
            algo_scores.append(scores)

        # [num_runs, num_tasks] – transpose, damit Tasks = Envs
        if algo_scores:
            stacked = np.vstack(algo_scores).T  # shape: (num_runs, num_tasks)
            score_dict[algo] = stacked

    return score_dict

def main():
    # Schritt 1: Rewards aggregieren
    df = aggregate_metrics.aggregate_reward_curves(BASE_PATH)

    # Schritt 2: In Score-Format überführen
    scores = prepare_score_dict(df, ENVS)

    # Schritt 3: IQM berechnen
    iqm_func = lambda x: np.array([metrics.aggregate_iqm(x)])
    iqm_estimates, iqm_cis = library.get_interval_estimates(
        scores,
        func=iqm_func,
        method='percentile',
        reps=5000,
        confidence_interval_size=0.95,
        task_bootstrap=False
    )

    print("\n Interquartilsmittel (IQM) mit 95%-Konfidenzintervall:")
    for method, val in iqm_estimates.items():
        ci = iqm_cis[method]
        print(f"{method}: {val[0]:.3f} (CI: {ci[0][0]:.3f} – {ci[1][0]:.3f})")

    # Schritt 4: Wahrscheinlichkeit der Verbesserung (A vs. B)
    methods = list(scores.keys())
    if len(methods) >= 2:
        a, b = methods[0], methods[1]

        prob_func = lambda x, y: np.array([metrics.probability_of_improvement(x, y)])
        prob_estimates, prob_cis = library.get_interval_estimates(
            {f'{a}_vs_{b}': [scores[a], scores[b]]},
            func=prob_func,
            method='percentile',
            reps=5000,
            confidence_interval_size=0.95,
        )

        prob = prob_estimates[f'{a}_vs_{b}'][0]
        ci = prob_cis[f'{a}_vs_{b}']
        print(f"\n🔀 Wahrscheinlichkeit, dass {a} besser ist als {b}: {prob:.3f} (CI: {ci[0][0]:.3f} – {ci[1][0]:.3f})")

if __name__ == "__main__":
    main()