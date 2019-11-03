# from import-stuff import *

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

n_rows = 6
n_cols = 7
maze = np.full((n_rows, n_cols), True)
maze[0:3, 2] = False
maze[4, 1:6] = False


class State:
    def __init__(self, index, value):
        self.index = index
        self.neighbours = []
        self.value = value

    def __str__(self):
        return str(self.index)

    def best_next_state(self):
        ne_values = array([ne.value for ne in self.neighbours])
        return self.neighbours[np.argmax(ne_values)]


def value_maze_from_states(states, n, m):
    vm = np.zeros((n, m))
    for state in states:
        vm[state.index] = state.value
    return vm

def policy_maze_from_states(states, n, m, ax):
    pass


states = []
for row in range(n_rows):
    for col in range(n_cols):
        if maze[row, col]:
            if row == 5 and col == 5:
                states.append(State(index=(row, col), value=1))    # target
            else:
                states.append(State(index=(row, col), value=0))

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

# for state in states:
#     print(state)
#     for ne in state.neighbours:
#         print('\t'+str(ne))
rewards = [1/(0.1+np.linalg.norm(np.array([state.index[0]-5, state.index[1]-5]))) for state in states]
decay = 0.97

n_iter = 4

fig, ax = plt.subplots(1, n_iter+1)
value_maze = value_maze_from_states(states, n_rows, n_cols)
ax[0].matshow(value_maze)

for i in range(1, n_iter+1):
    for state in states:
        ne_values = [ne.value for ne in state.neighbours]
        state.value = decay*np.sum(ne_values)
        ax[i].matshow(value_maze_from_states(states, n_rows, n_cols))
    print(value_maze_from_states(states, n_rows, n_cols))

plt.show()