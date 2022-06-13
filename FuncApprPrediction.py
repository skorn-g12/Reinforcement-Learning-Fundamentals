import numpy as np
import GridWorld
from collections import defaultdict
from sklearn.kernel_approximation import Nystroem, RBFSampler

# Let's have all the constants up front
states = [(0, 0), (0, 1), (0, 2), (0, 3),
          (1, 0), (1, 2), (1, 3),
          (2, 0), (2, 1), (2, 2), (2, 3)]

actions = {(0, 0): ("D", "R"), (0, 1): ("L", "R"), (0, 2): ("L", "D", "R"),
           (1, 0): ("D", "U"), (1, 2): ("D", "U", "R"),
           (2, 0): ("U", "R"), (2, 1): ("L", "R"), (2, 2): ("L", "U", "R"), (2, 3): ("L", "U")}

terminal_states = [(0, 3), (1, 3)]

permissible_initial_states = [(0, 0), (0, 1), (0, 2),
                              (1, 0), (1, 2),
                              (2, 0), (2, 1), (2, 2), (2, 3)]

policy = {}
for s in actions.keys():
    policy[s] = np.random.choice(actions[s])
"""
policy = {
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',

    (1, 0): 'U',
    (1, 2): 'U',

    (2, 0): 'U',
    (2, 1): 'R',
    (2, 2): 'U',
    (2, 3): 'L'
  }
"""
number_to_action = {"U": 0, "D": 1, "L": 2, "R": 3}
gamma = 0.9  # discount factor

trans_probs = {((0, 0), "R"): 0.5,
               ((0, 0), "D"): 0.5,

               ((0, 1), "L"): 0.45,
               ((0, 1), "R"): 0.55,

               ((0, 2), "L"): 0.6,
               ((0, 2), "R"): 0.3,
               ((0, 2), "D"): 0.1,

               ((1, 0), "U"): 0.6,
               ((1, 0), "D"): 0.4,

               ((1, 2), "U"): 0.6,
               ((1, 2), "R"): 0.2,
               ((1, 2), "D"): 0.2,

               ((2, 0), "U"): 0.6,
               ((2, 0), "R"): 0.4,

               ((2, 1), "L"): 0.4,
               ((2, 1), "R"): 0.6,

               ((2, 2), "L"): 0.3,
               ((2, 2), "R"): 0.4,
               ((2, 2), "U"): 0.3,

               ((2, 3), "U"): 0.5,
               ((2, 3), "L"): 0.5
               }

V = {}
for s in states:
    V[s] = 0

gamma = 0.9  # discount factor
eps = 0.1
ALPHA = 0.01

Q = defaultdict(list)

returns = defaultdict(list)

for s in permissible_initial_states:
    for a in actions[s]:
        Q[s, a] = 0
        returns[s, a] = []


def epsilon_greedy(s, policy):
    if np.random.random() < eps:  # Randomly choose an action
        a = np.random.choice(actions[s])
    else:
        a = policy[s]
    return a


def gather_samples(n_episodes=10000):
    samples = []
    for _ in range(n_episodes):
        # Start position and append
        # Play an episode
        start = (2, 0)
        samples.append(start)
        s = start
        while True:
            if s in terminal_states:
                break
            a = np.random.choice(actions[s])
            si, sj, r = grid.move(s, a)
            s2 = (si, sj)
            samples.append(s2)
            s = s2
    return samples


class Model:
    def __init__(self):
        samples = gather_samples()
        # self.featurizer = Nystroem()
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        self.w = np.zeros(dims)

    def predict(self, s):  # Will return Vhat(s)
        x = self.featurizer.transform([s])[0]
        return x @ self.w  # Dot product

    def phi_s(self, s):
        x = self.featurizer.transform([s])[0]
        return x


def experiment(model, grid, nEpisodes=5000):
    epoch = 0
    while epoch < nEpisodes:
        s = start
        while s not in terminal_states:
            a = epsilon_greedy(s, policy)
            si, sj, r = grid.move(s, a)
            Vs = model.predict(s)
            s2 = (si, sj)
            if s2 in terminal_states:
                target = r
            else:
                target = r + gamma * model.predict(s2)
            model.w += ALPHA * (target - Vs)*model.phi_s(s)
            s = s2
        epoch += 1
        print(epoch)


if __name__ == "__main__":
    start = (2, 0)
    rbf = RBFSampler()
    grid = GridWorld.GridWorld(start, states, actions, terminal_states)
    model = Model()
    experiment(model, grid)
    print("policy: ", policy)
    for s in states:
        if s in terminal_states:
            V[s] = 0
        else:
            V[s] = model.predict(s)
        print(s, ": ", V[s])

