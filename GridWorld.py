class GridWorld:
    def __init__(self, start, states, actions, terminal_states):
        self.start = start
        self.states = states
        self.actions = actions
        self.terminal_states = terminal_states
        self.i = start[0]
        self.j = start[1]

    def set_state(self, state):
        self.i = state[0]
        self.j = state[1]

    def get_reward(self, currState):
        pos_x = currState[0]
        pos_y = currState[1]
        pos = (pos_x, pos_y)
        r = -1
        if pos in self.terminal_states:
            if pos == self.terminal_states[0]:
                r = 5
            elif pos == self.terminal_states[1]:
                r = -5
        return r

    def move(self, currState, a):  # Move from currState to new state, s2 by performing action, a
        self.i = currState[0]
        self.j = currState[1]
        if a == "R":
            self.j += 1
        elif a == "L":
            self.j -= 1
        elif a == "U":
            self.i -= 1
        elif a == "D":
            self.i += 1
        r = self.get_reward((self.i, self.j))
        return self.i, self.j, r
