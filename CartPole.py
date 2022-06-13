import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
import gym

# Hyperparameter tuning
ALPHA = 0.1
GAMMA = 0.99


def epsilon_greedy(model, s, eps=0.1):
    p = np.random.random()
    if p < eps:  # Randomly choose an action
        return model.env.action_space.sample()
    else:
        values = model.predict_all_actions(s)
        return np.argmax(values)


def gather_samples(env, n_episodes=10000):
    samples = []
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            sa = np.concatenate((s, [a]))  # Action is binary, no need to one-hot encode it
            samples.append(sa)
            s, r, done, info = env.step(a)
    return samples


def test_agent(model, env, n_episodes=20):
    reward_per_episode = []
    for episode in range(n_episodes):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = epsilon_greedy(model, s, eps=0)
            s, r, done, info = env.step(a)
            episode_reward += r
        reward_per_episode.append(episode_reward)
    return np.mean(reward_per_episode)


def watch_agent(model, env, eps):
    s = env.reset()
    done = False
    episode_reward = 0
    while not done:
        a = epsilon_greedy(model, s, eps=eps)
        s, r, done, info = env.step(a)
        env.render()
        episode_reward += r
    print("Episode reward: ", episode_reward)


class Model:
    def __init__(self, env):
        self.env = env
        samples = gather_samples(env)
        # self.featurizer = Nystroem()
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        self.w = np.zeros(dims)

    def predict(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x @ self.w

    def predict_all_actions(self, s):
        return [self.predict(s, a) for a in range(self.env.action_space.n)]

    def grad(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    model = Model(env)
    reward_per_episode = []
    # watch_agent(model, env, eps=0)

    n_episodes = 2000
    for episode in range(n_episodes):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = epsilon_greedy(model, s)
            s2, r, done, info = env.step(a)
            target = 0
            if done:
                target = r
            else:
                values = model.predict_all_actions(s2)
                a_max = np.argmax(values)
                target = r + GAMMA * model.predict(s2, a_max)
            err = target - model.predict(s, a)
            model.w += ALPHA * err * model.grad(s, a)
            episode_reward += r
            s = s2
        # Print reward every 50th episode:
        if episode % 50 == 0:
            print(f" Episode: {episode}, Reward: {episode_reward}")

        reward_per_episode.append(episode_reward)

    test_reward = test_agent(model, env)
    print("Test agent avg reward: ", test_reward)
    plt.plot(reward_per_episode)
    plt.title("Reward per episode")
    plt.show()

    watch_agent(model, env, eps=0)
