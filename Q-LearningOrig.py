import numpy as np
import GridWorld
from collections import defaultdict

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


# Fixed policy
"""
policy = {
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'L': 1.0},

    (1, 0): {'U': 1.0},
    (1, 2): {'U': 1.0},

    (2, 0): {'U': 1.0},
    (2, 1): {'R': 1.0},
    (2, 2): {'U': 1.0},
    (2, 3): {'U': 1.0}
  }


policy = {
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'L',

    (1, 0): 'U',
    (1, 2): 'U',

    (2, 0): 'U',
    (2, 1): 'L',
    (2, 2): 'R',
    (2, 3): 'U'
  }
"""


policy = {}
for s in actions.keys():
    policy[s] = np.random.choice(actions[s])


number_to_action = {"U": 0, "D": 1, "L": 2, "R": 3}

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


def experiment():
    gamma = 0.9  # discount factor
    start = (2, 0)
    grid = GridWorld.GridWorld(start, states, actions, terminal_states)
    thr = 1e-5
    iter = 0
    alpha = 0.1
    while iter < 10000:  # Convergence loop
        s = start
        delta = 0
        while True:  # Game loop
            if s in terminal_states:
                break
            V_old = V[s]
            grid.set_state(s)

            a = epsilon_greedy(s, policy)
            si, sj, r = grid.move(s, a)
            s2 = (si, sj)

            if s2 not in terminal_states:
                # Get max_a
                k = [-500] * len(actions[s2])
                bCheck = 0
                for idx, a_new in enumerate(actions[s2]):
                    if Q[s2, a_new]:
                        k[idx] = (Q[s2, a_new])
                        bCheck = 1
                a_max = 0
                if bCheck == 1:
                    max_a_idx = np.argmax(k)
                    a_max = actions[s2][max_a_idx]
                else:
                    a_max = np.random.choice(actions[s2])
                Q[s, a] = Q[s, a] + alpha * ((r + gamma * Q[s2, a_max]) - Q[s, a])
            else:
                Q[s, a] = Q[s, a] + alpha * (r - Q[s, a])  # Target is 0
            s = s2
        iter += 1
        # print("iter : ", iter)


def print_value_for_policy(V):
    for s in states:
        print(" ", s, ": ", V[s])


if __name__ == "__main__":
    start = (2, 0)
    print("initial policy", policy)
    experiment()
    for s in permissible_initial_states:
        possible_actions = actions[s]
        argmaxMe = []
        for a in possible_actions:
            argmaxMe.append(Q[s, a])
        max_idx = np.argmax(argmaxMe)
        policy[s] = possible_actions[max_idx]

    print("Updated", policy)

