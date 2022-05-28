import numpy as np
import matplotlib.pyplot as plt

arm_probs = [0.2, 0.5, 0.7]
bandits = []

MAX_EPOCHS = 10000


class Bandit:
    def __init__(self, p, bOptimistic=0):
        self.p = p
        if bOptimistic == 0:
            self.p_est = 0
        elif bOptimistic == 1:
            self.p_est = 10
        else:
            self.p_est = 0
        self.N = 1

    def pull(self):
        val = np.random.random()
        return val < self.p  # Return reward

    def update_p_est(self, r):
        self.p_est = ((self.N - 1) * self.p_est + r) / self.N
        self.N += 1


def ucb(p_est, n, nj):
    return p_est + np.sqrt(2 * np.log(n) / nj)


if __name__ == "__main__":
    epsilons = [0, 0.01, 0.1, 0.5]

    explore_exploit_dict = {}
    eps = 0.1
    # Initialize a new bandit for each epsilon --> clean p_est
    for bOpt in [0, 1, 2]:
    # for bOpt in [0, 2]:
        bandit_0 = Bandit(arm_probs[0], bOpt)
        bandit_1 = Bandit(arm_probs[1], bOpt)
        bandit_2 = Bandit(arm_probs[2], bOpt)
        bandits = [bandit_0, bandit_1, bandit_2]
        rewards = np.zeros(MAX_EPOCHS)  # Initialize rewards to zeros for each epsilon
        N = 0
        if bOpt == 2:
            for idx, b in enumerate(bandits):
                arm_to_pull = idx
                r = b.pull()
                b.update_p_est(r)
                N += 1
        num_times_explored = 0
        num_times_exploited = 0

        for epoch in range(1, MAX_EPOCHS):
            arm_to_pull = 0
            N += 1
            if bOpt == 0:
                if np.random.random() < eps:  # Explore
                    arm_to_pull = np.random.randint(len(bandits))
                    num_times_explored += 1
                else:  # Exploit
                    arm_to_pull = np.argmax([b.p_est for b in bandits])
                    num_times_exploited += 1
            elif bOpt == 1:
                arm_to_pull = np.argmax([b.p_est for b in bandits])
                num_times_exploited += 1
            else:
                arm_to_pull = np.argmax([ucb(b.p_est, N, b.N) for b in bandits])
                num_times_exploited += 1
            r = bandits[arm_to_pull].pull()  # Pull arm and get reward
            bandits[arm_to_pull].update_p_est(r)  # update p_est
            rewards[epoch] = r  # Append current reward to rewards
        t = [(num_times_explored / (MAX_EPOCHS - 1)), (num_times_exploited / (MAX_EPOCHS - 1))]
        explore_exploit_dict[eps] = t  # amount of times explored, exploited : in percentage
        print("estimates: eps: ", eps)
        [print(b.p_est) for b in bandits]  # Print estimate according to this epsilon run

        plt.plot(np.cumsum(rewards) / (np.arange(MAX_EPOCHS) + 1))  # Plot cumsum

    print(explore_exploit_dict)
    plt.ylabel("rewards")
    plt.xlabel("Epochs")
    plt.legend(["epsilon greedy with 0.1", "Optimistic initial values(10)", "UCB 1"])
    # plt.legend(["epsilon greedy with 0.1", "UCB 1"])
    plt.show()
