import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1  # 10% change of exploring
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_est = 5.
        self.N = 1

    def pull(self):
        w = np.random.randn() + self.p
        return w

    def update(self, x):
        self.N += 1
        self.p_est = self.p_est + (1 / self.N) * (x - self.p_est)


def run_experiment(m1, m2, m3, eps):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_arm = np.argmax([b.p for b in bandits])
    nNumberOfBandits = len(bandits)
    for trial in range(NUM_TRIALS):
        if np.random.random() < eps:
            num_times_explored += 1
            nArmToPull = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            nArmToPull = np.argmax([b.p_est for b in bandits])

        if nArmToPull == optimal_arm:
            num_optimal += 1
        # Pull the arm for the bandit with the largest sample mean
        r = bandits[nArmToPull].pull()
        rewards[trial] = r
        bandits[nArmToPull].update(r)
    return bandits, rewards, num_optimal, num_times_explored, num_times_exploited


def print_stuff(bandits, rewards, num_optimal, num_times_explored, num_times_exploited):
    # print mean estimates for each bandit
    for b in bandits:
        print("mean estimate:", b.p_est)

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    return win_rates


if __name__ == "__main__":
    m1, m2, m3 = 1.5, 2.5, 3.5
    bandits, rewards, num_optimal, num_times_explored, num_times_exploited = run_experiment(m1, m2, m3, 0.1)
    c_01 = print_stuff(bandits, rewards, num_optimal, num_times_explored, num_times_exploited)

    bandits, rewards, num_optimal, num_times_explored, num_times_exploited = run_experiment(m1, m2, m3, 0.05)
    c_005 = print_stuff(bandits, rewards, num_optimal, num_times_explored, num_times_exploited)

    bandits, rewards, num_optimal, num_times_explored, num_times_exploited = run_experiment(m1, m2, m3, 0.01)
    c_001 = print_stuff(bandits, rewards, num_optimal, num_times_explored, num_times_exploited)

    # log scale plot
    plt.plot(c_01, label='eps = 0.1')
    plt.plot(c_005, label='eps = 0.05')
    plt.plot(c_001, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_01, label='eps = 0.1')
    plt.plot(c_005, label='eps = 0.05')
    plt.plot(c_001, label='eps = 0.01')
    plt.legend()
    plt.show()