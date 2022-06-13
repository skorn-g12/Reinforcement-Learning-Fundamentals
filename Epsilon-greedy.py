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
        else:
            self.p_est = 10
        self.N = 1

    def pull(self):
        val = np.random.random()
        return val < self.p  # Return reward

    def update_p_est(self, r):
        self.p_est = ((self.N - 1)*self.p_est + r) / self.N
        self.N += 1


if __name__ == "__main__":
    epsilons = [0, 0.01, 0.1, 0.5]
    plt.figure(figsize=(10, 6), tight_layout=True)
    explore_exploit_dict = {}
    for eps in epsilons:
        # Initialize a new bandit for each epsilon --> clean p_est
        bandit_0 = Bandit(arm_probs[0])
        bandit_1 = Bandit(arm_probs[1])
        bandit_2 = Bandit(arm_probs[2])
        bandits = [bandit_0, bandit_1, bandit_2]
        rewards = np.zeros(MAX_EPOCHS) # Initialize rewards to zeros for each epsilon

        num_times_explored = 0
        num_times_exploited = 0
        for epoch in range(1, MAX_EPOCHS):
            if np.random.random() < eps:  # Explore
                arm_to_pull = np.random.randint(len(bandits))
                num_times_explored += 1
            else:  # Exploit
                arm_to_pull = np.argmax([b.p_est for b in bandits])
                num_times_exploited += 1
            r = bandits[arm_to_pull].pull() # Pull arm and get reward
            bandits[arm_to_pull].update_p_est(r) # update p_est
            rewards[epoch] = r # Append current reward to rewards
        t = [(num_times_explored/(MAX_EPOCHS-1)), (num_times_exploited/(MAX_EPOCHS - 1)) ]
        explore_exploit_dict[eps] = t # amount of times explored, exploited : in percentage
        print("estimates: eps: ", eps)
        [print(b.p_est) for b in bandits]  # Print estimate according to this epsilon run
        print("explore exploit %", t)
        print("explore exploit %", num_times_explored, num_times_exploited)
        plt.plot(np.cumsum(rewards) / (np.arange(MAX_EPOCHS) + 1), '-')  # Plot cumsum

    print(explore_exploit_dict)
    plt.ylabel("Rewards")
    plt.xlabel("Epochs")
    # plt.legend([epsilons[0], epsilons[1], epsilons[2], epsilons[3]])
    plt.legend(
        ["eps={}".format(epsilons[0]), "eps ={}".format(epsilons[1]), "eps ={}".format(epsilons[2]),
         "eps={}".format(epsilons[3])])
    plt.title('Cumulative rewards for different values of epsilon')
    plt.show()

