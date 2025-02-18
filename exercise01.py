import numpy as np
import matplotlib.pyplot as plt

class StochasticBandit:
    def __init__(self, num_arms, bandit_type="bernoulli", means=None, random_init=True, delta=None):
        """
        Initialize a stochastic bandit with Gaussian or Bernoulli arms.

        Parameters:
        - num_arms (int): Number of arms.
        - bandit_type (str): "bernoulli" or "gaussian".
        - means (list or None): Optional predefined means for arms.
        - random_init (bool): If True, means are randomly generated.
        - delta (float or None): Reward gap Δ for mean adjustment.
        """
        self.num_arms = num_arms
        self.bandit_type = bandit_type.lower()
        self.means = np.array(means) if means is not None else None
        self.delta = delta
        
        if random_init or means is None:
            self._initialize_means()
        
    def _initialize_means(self):
        """Initializes arm means based on bandit type."""
        if self.bandit_type == "bernoulli":
            self.means = np.random.uniform(0, 1, self.num_arms)
        elif self.bandit_type == "gaussian":
            self.means = np.random.randn(self.num_arms)  # Standard normal distribution
        else:
            raise ValueError("Invalid bandit type. Use 'bernoulli' or 'gaussian'.")
        
        if self.delta is not None:
            self._apply_reward_gap()

    def _apply_reward_gap(self):
        """Adjusts means according to the reward gap Δ."""
        sorted_indices = np.argsort(self.means)[::-1]  # Sort in descending order
        highest_mean = self.means[sorted_indices[0]]
        for k in range(self.num_arms):
            new_mean = highest_mean - k * self.delta
            if self.bandit_type == "bernoulli" and new_mean < 0:
                new_mean = 0  # Ensure valid probabilities
            self.means[sorted_indices[k]] = new_mean
    
    def pull_arm(self, arm_index):
        """Simulates pulling an arm and returns a reward."""
        if self.bandit_type == "bernoulli":
            return np.random.binomial(1, self.means[arm_index])
        elif self.bandit_type == "gaussian":
            return np.random.normal(self.means[arm_index], 1)
        else:
            raise ValueError("Invalid bandit type.")
    
    def get_means(self):
        """Returns the current means of the bandit arms."""
        return self.means

class ETCAlgorithm:
    def __init__(self, bandit, exploration_rounds):
        """
        Implements the Exploration-Then-Commit (ETC) algorithm.
        
        Parameters:
        - bandit (StochasticBandit): The bandit instance to play.
        - exploration_rounds (int): Number of rounds to explore before committing.
        """
        self.bandit = bandit
        self.exploration_rounds = exploration_rounds
        self.total_rounds = 0
        self.rewards = np.zeros(bandit.num_arms)
        self.counts = np.zeros(bandit.num_arms, dtype=int)
        self.best_arm = None
    
    def step(self):
        """Executes one step of the ETC algorithm."""
        if self.total_rounds < self.exploration_rounds * self.bandit.num_arms:
            # Exploration phase: cycle through each arm
            arm = self.total_rounds % self.bandit.num_arms
        else:
            # Exploitation phase: commit to the best estimated arm
            if self.best_arm is None:
                self.best_arm = np.argmax(self.rewards / self.counts)
            arm = self.best_arm
        
        reward = self.bandit.pull_arm(arm)
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.total_rounds += 1
        
        return arm, reward

def run_simulation(n_runs=1000, time_horizon=10000, exploration_rounds=100):
    num_arms = 10
    regrets = np.zeros((n_runs, time_horizon))
    correct_action_rates = np.zeros((n_runs, time_horizon))
    arm_selection_counts = np.zeros((n_runs, num_arms, time_horizon))
    
    for run in range(n_runs):
        bandit = StochasticBandit(num_arms=num_arms, bandit_type="gaussian")
        etc = ETCAlgorithm(bandit, exploration_rounds)
        best_arm = np.argmax(bandit.get_means())
        
        for t in range(time_horizon):
            chosen_arm, reward = etc.step()
            regrets[run, t] = np.max(bandit.get_means()) - bandit.get_means()[chosen_arm]
            correct_action_rates[run, t] = (chosen_arm == best_arm)
            arm_selection_counts[run, chosen_arm, t] += 1
    
    return regrets.mean(axis=0), correct_action_rates.mean(axis=0), arm_selection_counts.mean(axis=0)

def plot_results(regret, correct_action_rate, arm_selection_probs):
    plt.figure(figsize=(12, 8))
    
    # Regret over time
    plt.subplot(2, 2, 1)
    plt.plot(regret)
    plt.xlabel("Time step")
    plt.ylabel("Regret")
    plt.title("Regret over time")
    
    # Correct action rate over time
    plt.subplot(2, 2, 2)
    plt.plot(correct_action_rate)
    plt.xlabel("Time step")
    plt.ylabel("Correct Action Rate")
    plt.title("Correct Action Rate over time")
    
    # Arm selection probabilities over time
    plt.subplot(2, 2, 3)
    for arm in range(arm_selection_probs.shape[0]):
        plt.plot(arm_selection_probs[arm], label=f"Arm {arm}")
    plt.xlabel("Time step")
    plt.ylabel("Selection Probability")
    plt.title("Arm Selection Probabilities")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    regret, correct_action_rate, arm_selection_probs = run_simulation()
    plot_results(regret, correct_action_rate, arm_selection_probs)
