import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curves(returns_dict):
    """
    Plots the learning curves of different algorithms.

    Args:
        returns_dict (dict): A dictionary where keys are algorithm names and values are lists of returns.
    """
    plt.figure(figsize=(12, 6))
    for algo_name, returns in returns_dict.items():
        plt.plot(returns, label=algo_name)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()


# CSV-Dateien einlesen
reinforce_df = pd.read_csv('exercise10/algorithms/returns/reinforce_returns.csv')
minibatch_df = pd.read_csv('exercise10/algorithms/returns/minibatch_reinforce_returns.csv')

# Dictionary für die Visualisierung vorbereiten
returns_dict = {
    'REINFORCE': reinforce_df['return'],
    'Minibatch REINFORCE': minibatch_df['return']
}

# Plotten
plot_learning_curves(returns_dict)
