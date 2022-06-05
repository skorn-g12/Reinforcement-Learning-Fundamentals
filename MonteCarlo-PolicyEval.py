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


returns = defaultdict(list)
for s in states:
    returns[s] = []

"""
# Probabilistic policy
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


def play_episode(max_steps, grid):
    step = 1
    currState = permissible_initial_states[np.random.choice(len(permissible_initial_states))]
    visited_states = [currState]
    rewards = [0]
    print("initial state: ", currState)
    while step <= max_steps:
        grid.set_state(currState)
        a = policy[currState]  # Get action based on policy
        si, sj, r = grid.move(currState, a)  # Move from currState to new state, s2 by performing action, a.
        s2 = (si, sj)
        rewards.append(r)
        visited_states.append(s2)
        step += 1
        if s2 in terminal_states:
            break
        currState = s2
    return visited_states, rewards


if __name__ == "__main__":
    iter = 0
    start = (2, 0)
    # print("initial policy", policy)
    grid = GridWorld.GridWorld(start, states, actions, terminal_states)

    max_number_of_steps_per_episode = 20
    while True:
        iter += 1
        visited_states, rewards = play_episode(max_number_of_steps_per_episode, grid)
        G = 0
        for t in range((len(visited_states) - 2), -1, -1):
            if t == 0:
                break
            G = rewards[t+1] + gamma*G
            s = visited_states[t]
            if visited_states[t] not in visited_states[:t]:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
        if iter >= 100:
            break

    print(V)