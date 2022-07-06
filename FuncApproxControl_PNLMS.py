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


def get_a_max(s, model):
    k = [-500] * len(actions[s])
    bCheck = 0
    a_max = 0
    for idx, a_new in enumerate(actions[s]):
        k[idx] = model.compute_Qsa(s, a_new)
    max_a_idx = np.argmax(k)
    a_max = actions[s][max_a_idx]
    a = a_max
    return a


def epsilon_greedy(s, model):
    if np.random.random() < eps:  # Randomly choose an action
        a = np.random.choice(actions[s])
    else:

        a = get_a_max(s, model)
    return a


def gather_samples(n_episodes=10000):
    samples = []
    for _ in range(n_episodes):
        # Start position and append
        # Play an episode
        start = (2, 0)
        s = start
        a = np.random.choice(actions[s])
        a_one_hot = one_hot_encoder(a)

        while True:
            if s in terminal_states:
                break
            vect = [s[0], s[1], a_one_hot[0], a_one_hot[1], a_one_hot[2], a_one_hot[3]]
            samples.append(vect)

            si, sj, r = grid.move(s, a)
            s2 = (si, sj)
            s = s2
            a = 0
            if s not in terminal_states:
                a = np.random.choice(actions[s])
                a_one_hot = one_hot_encoder(a)
    return samples


def one_hot_encoder(a):
    if a == "U":
        return [1, 0, 0, 0]
    elif a == "D":
        return [0, 1, 0, 0]
    elif a == "L":
        return [0, 0, 1, 0]
    elif a == "R":
        return [0, 0, 0, 1]


class Model:
    def __init__(self):
        samples = gather_samples()
        # self.featurizer = Nystroem()
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        self.w = np.zeros(dims)

    def predict_V(self, s):  # Will return Vhat(s)
        x = self.featurizer.transform([s])[0]
        return x @ self.w  # Dot product

    def compute_Qsa(self, s, a):  #
        one_hot_a = one_hot_encoder(a)
        vect = [s[0], s[1], one_hot_a[0], one_hot_a[1], one_hot_a[2], one_hot_a[3]]
        x = self.featurizer.transform([vect])[0]
        return x @ self.w  # Dot product

    def phi_s_a(self, s, a):
        one_hot_a = one_hot_encoder(a)
        vect = [s[0], s[1], one_hot_a[0], one_hot_a[1], one_hot_a[2], one_hot_a[3]]
        # vect = [s[0], s[1], a]
        x = self.featurizer.transform([vect])[0]
        return x


def weightUpdateEq(s, a, w, error, model):
    one_hot_a = one_hot_encoder(a)
    vect = [s[0], s[1], one_hot_a[0], one_hot_a[1], one_hot_a[2], one_hot_a[3]]
    x = model.featurizer.transform([vect])[0]
    res = sum(map(lambda i: i * i, x))
    variance = res / len(x)
    l_inf = max(np.abs(w))
    rho = 1e-3
    delta = 1e-2
    l_inf_p = max(delta, l_inf)
    g = np.zeros(len(w))
    for i in range(w.shape[0]):
        g[i] = max(rho * l_inf_p, np.abs(w[i]))
    g_bar = np.sum(g) / len(g)
    w = w + ALPHA * (np.multiply(g, x) * error) / (variance * g_bar * len(w))
    return w


def experiment(model, grid, nEpisodes=5000):
    epoch = 0
    while epoch < nEpisodes:
        s = start
        while s not in terminal_states:
            a = epsilon_greedy(s, model)
            si, sj, r = grid.move(s, a)
            Qsa = model.compute_Qsa(s, a)
            # Q[s, a] = Qsa
            s2 = (si, sj)
            a2 = 0
            if s2 in terminal_states:
                target = r
            else:
                amax = get_a_max(s2, model)
                Qs2a2 = model.compute_Qsa(s2, amax)
                # Q[s2, a2] = Qs2a2
                target = r + gamma * Qs2a2
                a2 = epsilon_greedy(s2, model)
            error = target - Qsa
            model.w = weightUpdateEq(s, a, model.w, error, model)
            # model.w += ALPHA * (target - Qsa) * model.phi_s_a(s, a)
            s = s2
            a = a2
        epoch += 1
        print(epoch)


if __name__ == "__main__":
    start = (2, 0)
    rbf = RBFSampler()
    print("initial policy: ", policy)
    grid = GridWorld.GridWorld(start, states, actions, terminal_states)
    model = Model()
    experiment(model, grid)

    for s in permissible_initial_states:
        possible_actions = actions[s]
        argmaxMe = []
        for a in possible_actions:
            Q[s, a] = model.compute_Qsa(s, a)
            argmaxMe.append(Q[s, a])
        max_idx = np.argmax(argmaxMe)
        policy[s] = possible_actions[max_idx]
    print("updated policy: ", policy)
