# from p1 import *


import numpy as np

n_rows = 6
n_cols = 7
maze = np.full((n_rows, n_cols), True)
maze[0:3, 2] = False
maze[4, 1:6] = False

states = []
for row in range(n_rows):
    for col in range(n_cols):
        if maze[row, col]:
            states.append((row, col))

n_states = len(states)
p = np.zeros((n_states, n_states))

ne_rel_inds = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
neighbours = [None for _ in range(n_states)]
for state in states:
    ne_states = []
    for ne_rel_ind in ne_rel_inds:

        ne_ind = (state[0] + ne_rel_ind[0], state[1] + ne_rel_ind[1])

        if (n_rows > ne_ind[0] >= 0) and (n_cols > ne_ind[1] >= 0):
            if maze[ne_ind]:
                ne_states.append(ne_ind)
    neighbours[states.index(state)] = ne_states

    n_nes = len(ne_states)

    for ne in ne_states:
        p[states.index(state), states.index(ne)] = 1/n_nes

rewards = [1/(0.1+np.linalg.norm(np.array([state[0]-5, state[1]-5]))) for state in states]

decay = 0.9

def value(states, rewards, val)
    


