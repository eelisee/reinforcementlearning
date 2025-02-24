import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

class StochasticBandit:
    def __init__(self, n_arms, bandit_type='gaussian', means=None, reward_gap=None):
        self.n_arms = n_arms
        self.bandit_type = bandit_type
        self.reward_gap = reward_gap

        if means is not None:
            self.means = means
        else:
            if bandit_type == 'gaussian':
                self.means = np.random.randn(n_arms)
            elif bandit_type == 'bernoulli':
                self.means = np.random.rand(n_arms)
            else:
                raise ValueError("Unsupported bandit type. Use 'gaussian' or 'bernoulli'.")

        if reward_gap is not None:
            self._apply_reward_gap()

    def _apply_reward_gap(self):
        sorted_indices = np.argsort(self.means)[::-1]
        mu_star = self.means[sorted_indices[0]]
        for k in range(1, self.n_arms):
            new_mean = mu_star - k * self.reward_gap
            if self.bandit_type == 'bernoulli' and new_mean < 0:
                new_mean = 0
            self.means[sorted_indices[k]] = new_mean

    def pull(self, arm):
        if arm < 0 or arm >= self.n_arms:
            raise ValueError("Arm index out of range.")
        
        if self.bandit_type == 'gaussian':
            return np.random.normal(self.means[arm], 1)
        elif self.bandit_type == 'bernoulli':
            return np.random.binomial(1, self.means[arm])
        else:
            raise ValueError("Unsupported bandit type. Use 'gaussian' or 'bernoulli'.")

class ETCAlgorithm:
    def __init__(self, bandit, exploration_rounds):
        self.bandit = bandit
        self.exploration_rounds = exploration_rounds
        self.arm_counts = np.zeros(bandit.n_arms)
        self.arm_rewards = np.zeros(bandit.n_arms)
        self.total_pulls = 0
        self.best_arm = None
        self.regrets = []
        self.correct_action_rates = []
        self.arm_selection_probs = np.zeros(bandit.n_arms)
        self.estimated_means = np.zeros((bandit.n_arms, 0))

    def step(self):
        if self.total_pulls < self.bandit.n_arms * self.exploration_rounds:
            # Exploration phase
            arm = self.total_pulls % self.bandit.n_arms
            reward = self.bandit.pull(arm)
            self.arm_counts[arm] += 1
            self.arm_rewards[arm] += reward
            self.total_pulls += 1
        else:
            # Exploitation phase
            if self.best_arm is None:
                self.best_arm = np.argmax(self.arm_rewards / self.arm_counts)
            arm = self.best_arm
            reward = self.bandit.pull(arm)
        
        # Update regrets and correct action rates
        optimal_reward = self.bandit.means.max()
        regret = optimal_reward - reward
        self.regrets.append(regret)
        self.correct_action_rates.append(1 if arm == np.argmax(self.bandit.means) else 0)
        
        # Update arm selection probabilities
        self.arm_selection_probs[arm] += 1
        
        # Update estimated means
        estimated_means = self.arm_rewards / np.maximum(self.arm_counts, 1)
        self.estimated_means = np.column_stack((self.estimated_means, estimated_means))
        
        return reward

    def get_average_arm_selection_probs(self, total_steps):
        return self.arm_selection_probs / total_steps

def run_experiment(n_arms, reward_gap, exploration_rounds, n_steps):
    bandit = StochasticBandit(n_arms, bandit_type='gaussian', reward_gap=reward_gap)
    etc = ETCAlgorithm(bandit, exploration_rounds)
    regrets = np.zeros(n_steps)
    correct_action_rates = np.zeros(n_steps)
    for t in range(n_steps):
        etc.step()
        regrets[t] = np.sum(etc.regrets)
        correct_action_rates[t] = np.mean(etc.correct_action_rates)
    arm_selection_probs = etc.get_average_arm_selection_probs(n_steps)
    return regrets, correct_action_rates, arm_selection_probs, etc.estimated_means

if __name__ == '__main__':
    # Experiment parameters
    n_arms = 10
    reward_gap = 0.1
    exploration_rounds = 10
    n_steps = 10000
    n_iterations = 1000

    # Initialize arrays to store results
    all_regrets = np.zeros((n_iterations, n_steps))
    all_correct_action_rates = np.zeros((n_iterations, n_steps))
    all_arm_selection_probs = np.zeros((n_iterations, n_arms))
    all_estimated_means = np.zeros((n_iterations, n_arms, n_steps))

    # Run experiments in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, n_arms, reward_gap, exploration_rounds, n_steps) for _ in range(n_iterations)]
        for i, future in enumerate(futures):
            print(f"Processing iteration {i+1}/{n_iterations}")
            regrets, correct_action_rates, arm_selection_probs, estimated_means = future.result()
            all_regrets[i] = regrets
            all_correct_action_rates[i] = correct_action_rates
            all_arm_selection_probs[i] = arm_selection_probs
            all_estimated_means[i] = estimated_means

    # Calculate averages
    avg_regrets = np.mean(all_regrets, axis=0)
    avg_correct_action_rates = np.mean(all_correct_action_rates, axis=0)
    avg_arm_selection_probs = np.mean(all_arm_selection_probs, axis=0)
    avg_estimated_means = np.mean(all_estimated_means, axis=0)

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot (a) the regrets over time
    plt.subplot(2, 2, 1)
    plt.plot(avg_regrets)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title('Regrets over Time')

    # Plot (b) the correct action rates over time
    plt.subplot(2, 2, 2)
    plt.plot(avg_correct_action_rates)
    plt.xlabel('Time')
    plt.ylabel('Correct Action Rate')
    plt.title('Correct Action Rates over Time')

    # Plot (c) the estimations of the means of the different arms
    plt.subplot(2, 2, 3)
    for arm in range(n_arms):
        plt.plot(avg_estimated_means[arm] / np.mean(all_estimated_means[:, arm, :], axis=0), label=f'Arm {arm}')
    plt.xlabel('Time')
    plt.ylabel('Estimated Mean / Actual Mean')
    plt.title('Estimations of the Means of the Arms')
    plt.legend()

    # Plot (d) the average probabilities of choosing each of the arms over time
    plt.subplot(2, 2, 4)
    plt.bar(range(n_arms), avg_arm_selection_probs)
    plt.xlabel('Arm')
    plt.ylabel('Average Probability')
    plt.title('Average Probabilities of Choosing Each Arm')

    plt.tight_layout()
    plt.show()