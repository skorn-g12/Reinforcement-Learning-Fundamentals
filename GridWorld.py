class GridWorld:
    def __init__(self, start, states, actions, terminal_states):
        self.i = start[0]
        self.j = start[1]
        self.states = states
        self.actions = actions
        self.terminal_states = terminal_states

    def get_reward(self, pos):
        if (pos[0], pos[1]) == self.terminal_states[0]:  # Win
            r = 5
        elif (pos[0], pos[1]) == self.terminal_states[1]:  # Lose
            r = -5
        else:
            r = -1
        return r

    def get_next_state(self, pos, a):
        i = pos[0]
        j = pos[1]
        if a in self.actions[(pos[0], pos[1])]:
            if a == "U":
                i -= 1
            elif a == "D":
                i += 1
            elif a == "L":
                j -= 1
            elif a == "R":
                j += 1
            else:
                print("Action not in action space")
        return i, j

    def set_state(self, pos):
        self.i = pos[0]
        self.j = pos[1]

    def move(self, pos, a):
        if a in self.actions[(pos[0], pos[1])]:
            if a == "U":
                self.i -= 1
            elif a == "D":
                self.i += 1
            elif a == "L":
                self.j -= 1
            elif a == "R":
                self.j += 1
            else:
                print("Action not in action space")
        r = self.get_reward((self.i, self.j))
        return self.i, self.j, r
