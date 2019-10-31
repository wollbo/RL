# from import-stuff import *

import numpy as np

n_rows = 6
n_cols = 7
maze = np.full((n_rows, n_cols), True)
maze[0:3, 2] = False
maze[4, 1:6] = False


class State:
    def __init__(self, index):
        self.index = index
        self.neighbours = []

    def __str__(self):
        return str(self.index)


states = []
for row in range(n_rows):
    for col in range(n_cols):
        if maze[row, col]:
            states.append(State(index=(row, col)))

n_states = len(states)
p = np.zeros((n_states, n_states))

for state in states:
    state.neighbours.append(state)
    for other_state in states:
        if np.abs(other_state.index[0]-state.index[0]) == 1 and np.abs(other_state.index[1]-state.index[1]) == 0:
            state.neighbours.append(other_state)
        elif np.abs(other_state.index[0]-state.index[0]) == 0 and np.abs(other_state.index[1]-state.index[1]) == 1:
            state.neighbours.append(other_state)

for state in states:
    n_nes = len(state.neighbours)
    for ne in state.neighbours:
        p[states.index(state), states.index(ne)] = 1/n_nes

print(p[:10, :10])

for state in states:
    print(state)
    for ne in state.neighbours:
        print('\t'+str(ne))


rewards = [1/(0.1+np.linalg.norm(np.array([state.index[0]-5, state.index[1]-5]))) for state in states]

decay = 0.9


def q(current_state, neighbours, rewards, decay):
    nes = neighbours[states.index(current_state)]
    return [rewards[states.index(ne)] + decay * q(ne, neighbours, rewards, decay) for ne in nes]


#print(q((0, 0), neighbours, rewards, decay))


