# from import-stuff import *

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

n_rows = 6
n_cols = 7
maze = np.full((n_rows, n_cols), True)
maze[0:3, 2] = False
maze[4, 1:6] = False


class State:
    def __init__(self, index):
        self.index = index
        self.neighbours = []
        self.value = 0
        self.policy = None  # index of neighbour to go to!

    def __str__(self):
        return 'state: ' + str(self.index) +\
               ', value: ' + str(self.value) +\
               ', next: ' + str(self.neighbours[self.policy])


def value_maze_from_states(states, n, m):
    vm = np.zeros((n, m))
    for state in states:
        vm[state.index] = state.value
    return vm


def policy_maze_from_states(states):
    for state in states:
        other_state = state.neighbours[state.policy]
        x = state.index[1]
        dx = other_state.index[1] - x
        y = -state.index[0]
        dy = -other_state.index[0] - y
        plt.arrow(x, y, dx, dy, fc='k', ec='k', head_width=0.15, head_length=0.4, length_includes_head=True)
    plt.gca().set_aspect('equal')
    plt.axis([-1, 7, -6, 1])


states = []
for row in range(n_rows):
    for col in range(n_cols):
        if maze[row, col]:
            if row == 5 and col == 5:
                target = State(index=(row, col))
                states.append(target)
            else:
                states.append(State(index=(row, col)))
n_states = len(states)

for state in states:
    state.neighbours.append(state)
    for other_state in states:
        if np.abs(other_state.index[0]-state.index[0]) == 1 and np.abs(other_state.index[1]-state.index[1]) == 0:
            state.neighbours.append(other_state)
        elif np.abs(other_state.index[0]-state.index[0]) == 0 and np.abs(other_state.index[1]-state.index[1]) == 1:
            state.neighbours.append(other_state)

transition_probability = np.zeros((n_states, n_states))
for state in states:
    n_nes = len(state.neighbours)
    for ne in state.neighbours:
        transition_probability[states.index(state), states.index(ne)] = 1/n_nes

transition_reward = np.zeros((n_states, n_states))
nes = target.neighbours
for ne in nes:
    transition_reward[states.index(ne), states.index(target)] = 10   # reward for going to / staying at the target!

gamma = 0.99

n_iter = 10
fig, ax = plt.subplots(1, n_iter)
fig1 = plt.figure(1)

for i in range(n_iter):
    " VALUE UPDATE "
    for state in states:
        value_elements = [transition_reward[states.index(state), states.index(ne)] + gamma*ne.value for ne in state.neighbours]
        state.value = sum(value_elements)

    " POLICY UPDATE "
    for state in states:
        value_elements = [transition_reward[states.index(state), states.index(ne)] + gamma*ne.value for ne in state.neighbours]
        state.policy = np.argmax(array([value_elements]))

    # print(value_maze_from_states(states, n_rows, n_cols))
    # ax[i].matshow(value_maze_from_states(states, n_rows, n_cols))

    plt.figure(1)
    plt.subplot(1, n_iter, i+1)
    policy_maze_from_states(states)

plt.show()




