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

policy = {}
for s in actions.keys():
    policy[s] = np.random.choice(actions[s])

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


def EvaluatePolicy(policy, grid):
    thr = 1e-5
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
                # action_prob = policy[s][a]
                # transition_prob = trans_probs[s,a]
                # new_v = new_v + action_prob*transition_prob*(r + gamma * V[s2])
                new_v = new_v + (r + gamma * V[s2])
                k = (r + gamma * V[(si, sj)])
            V[s] = new_v
            delta = max(delta, np.abs(V_old - V[s]))
        if delta < thr:
            break
        print("iter : ", iter)


def ImprovePolicy(policy, grid):
    bPolicy = 1
    for s in states:
        if s in terminal_states:
            continue
        a_old = policy[s]
        maxThisGuy = []
        for a in actions[s]:
            grid.set_state(s)
            si, sj, r = grid.move(s, a)
            s2 = (si, sj)
            maxThisGuy.append(trans_probs[s, a] * (r + gamma * V[s2]))
        a_new_idx = np.argmax(maxThisGuy)
        a_new = actions[s][a_new_idx]
        if a_new != a_old: # If any state's action deviates from the policy, policy is not converged
            bPolicy = 0
            policy[s] = a_new
    return bPolicy


if __name__ == "__main__":
    iter = 0
    start = (2, 0)
    print("initial policy", policy)
    grid = GridWorld.GridWorld(start, states, actions, terminal_states)
    while True:
        EvaluatePolicy(policy, grid)
        bPolicy = ImprovePolicy(policy, grid)
        if bPolicy == 1:
            break
        iter += 1

    print(iter)
    print("updated", policy)