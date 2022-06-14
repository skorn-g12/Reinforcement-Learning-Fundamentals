import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from gym import wrappers
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
import matplotlib


class FeatureTransformer:
    def __init__(self, env, n_components=500):
        samples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(samples)  # Z scoring observations
        featurizer = FeatureUnion([("rbf1", RBFSampler(gamma=5, n_components=n_components)),
                                   ("rbf2", RBFSampler(gamma=2, n_components=n_components)),
                                   ("rbf3", RBFSampler(gamma=1, n_components=n_components)),
                                   ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))])
        example_features = featurizer.fit_transform(scaler.transform(samples))
        self.dims = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])  # Target 0 --> Opt initial values
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps=0.1):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_one(model, env, gamma, eps=0.1):
    observation = env.reset()
    s = observation
    done = False
    total_reward = 0
    iters = 0
    while not done and iters < 10000:
        a = model.sample_action(s, eps=eps)
        s2, r, done, info = env.step(a)

        next = model.predict(s2)
        G = r + gamma * np.max(next[0])
        model.update(s, a, G)
        total_reward += r
        s = s2
        iters += 1
        # env.render()
    return total_reward


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                           rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99

    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1 * (0.97 ** n)
        if n == 199:
            print("eps:", eps)
        totalreward = play_one(model, env, gamma, eps)
        totalrewards[n] = totalreward
        if (n + 1) % 100 == 0:
            print("episode:", n, "total reward:", totalreward)
        print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
        print("total steps:", -totalrewards.sum())

    if True:
        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()

        plot_cost_to_go(env, model)
        plot_running_avg(totalrewards)
