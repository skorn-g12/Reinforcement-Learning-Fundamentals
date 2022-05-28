import numpy as np
import GridWorld

# Let's have all the constants up front
states = [(0, 0), (0, 1), (0, 2), (0, 3),
          (1, 0), (1, 2), (1, 3),
          (2, 0), (2, 1), (2, 2), (2, 3)]

actions = {(0, 0): ("D", "R"), (0, 1): ("L", "R"), (0, 2): ("L", "D", "R"),
           (1, 0): ("D", "U"), (1, 2): ("D", "U", "R"),
           (2, 0): ("U", "R"), (2, 1): ("L", "R"), (2, 2): ("L", "U", "R"), (2, 3): ("L", "U")}

terminal_states = [(0, 3), (1, 3)]

# Fixed policy
"""
policy = {
    (0, 0): 'R', (0, 1): 'R', (0, 2): 'R',
    (1, 0): 'U',              (1, 2): 'U',
    (2, 0): 'U', (2, 1): 'R', (2, 2): 'U', (2, 3): 'U'}
"""

# Probabilistic policy
"""
policy = {
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'R': 1.0},

    (1, 0): {'U': 1.0},
    (1, 2): {'U': 0.6, 'R': 0.3, 'D': 0.1},

    (2, 0): {'U': 0.5, 'R': 0.5},
    (2, 1): {'R': 1.0},
    (2, 2): {'U': 1.0},
    (2, 3): {'L': 1.0}
  }
"""
policy = {
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'R': 1.0},

    (1, 0): {'U': 1.0},
    (1, 2): {'U': 1.0},

    (2, 0): {'U': 1.0},
    (2, 1): {'R': 1.0},
    (2, 2): {'U': 1.0},
    (2, 3): {'L': 1.0}
  }

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


def experiment(V):
    gamma = 0.9  # discount factor
    start = (2, 0)
    grid = GridWorld.GridWorld(start, states, actions, terminal_states)
    thr = 1e-5
    iter = 0
    while True:
        delta = 0
        for s in states:  # For every state
            V_old = V[s]
            new_v = 0
            if s in terminal_states:
                continue
            for a in policy[s]:  # For the action specified in the policy
                grid.set_state(s)
                si, sj, r = grid.move(s, a)
                s2 = (si, sj)
                action_prob = policy[s][a]
                # transition_prob = trans_probs[s,a]
                # new_v = new_v + action_prob*transition_prob*(r + gamma * V[s2])
                new_v = new_v + action_prob * (r + gamma * V[s2])
                k = (r + gamma * V[(si, sj)])
            V[s] = new_v
            delta = max(delta, np.abs(V_old - V[s]))
        if delta < thr:
            break
        iter += 1
        print("iter : ", iter)


def print_value_for_policy(V):
    for s in states:
        print(" ", s, ": ", V[s])


if __name__ == "__main__":
    experiment(V)
    print_value_for_policy(V)
