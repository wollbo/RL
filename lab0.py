# from import-stuff import *

import numpy as np
from numpy import array
from numpy import nan
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

n_rows = 6
n_cols = 7
maze = np.full((n_rows, n_cols), True)
maze[0:3, 2] = False
maze[4, 1:6] = False

# maze[7, 3:7] = False
# maze[7:9, 3] = False
# maze[7:9, 7] = False


class State:
    def __init__(self, index):
        self.index = index
        self.neighbours = []
        self.value = 0
        self.next_value = 0
        self.policy = 0         # index of neighbour to go to!
        self.next_policy = 0

    def __str__(self):
        return 'state: ' + str(self.index) +\
               ', value: ' + str(self.value) +\
               ', next: ' + str(self.neighbours[self.policy])

    def update_value(self):
        no_change = np.abs(self.value - self.next_value) <= 1e-2        # check convergence
        self.value = self.next_value                                    # update value
        return no_change

    def update_policy(self):
        no_change = self.policy == self.next_policy     # check convergence
        self.policy = self.next_policy                  # update policy
        return no_change


def get_state_values(states, n, m):
    vm = np.full((n, m), nan)
    for state in states:
        if state.value != 0:
            vm[state.index] = state.value
    return vm


def show_state_policies(states):
    # plt.subplot(n, m, i) must come before calling this function
    ax2.clear()
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    for state in states:
        other_state = state.neighbours[state.policy]
        x = state.index[1]
        dx = other_state.index[1] - x
        y = -state.index[0]
        dy = -other_state.index[0] - y
        plt.arrow(x, y, dx, dy, fc='k', ec='k', head_width=0.25, head_length=0.5, length_includes_head=True)


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
    transition_reward[states.index(ne), states.index(target)] = 100   # reward for going to / staying at the target!

gamma = 0.9

policy_convergence = np.full(n_states, False)
value_convergence = np.full(n_states, False)

i = 0

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

while not (all(policy_convergence) and all(value_convergence)):
    print('i={:.0f}'.format(i))

    " VALUE UPDATE "
    for state in states:
        value_elements = array([transition_probability[states.index(state), states.index(ne)] *
                                (transition_reward[states.index(state), states.index(ne)] +
                                 gamma*ne.value) for ne in state.neighbours])
        state.next_value = np.sum(value_elements)
    for state in states:
        value_convergence[states.index(state)] = state.update_value()
    if all(value_convergence):
        print('\tvalues: converged')
    else:
        print('\tvalues: not converged')

    " POLICY UPDATE "
    for state in states:
        value_elements = [transition_reward[states.index(state), states.index(ne)] +
                          gamma*ne.value for ne in state.neighbours]
        state.next_policy = np.argmax(array([value_elements]))
    for state in states:
        policy_convergence[states.index(state)] = state.update_policy()
    if all(policy_convergence):
        print('\tpolicies: converged')
    else:
        print('\tpolicies: not converged')

    values = get_state_values(states, n_rows, n_cols)
    print(values)

    ax1.matshow(np.log(values))
    ax1.set_xlabel('State values')
    show_state_policies(states)
    ax2.set_xlabel('State policies')
    plt.gca().set_aspect('equal')
    plt.axis([-1, n_cols, -n_rows, 1])
    plt.pause(0.01)

    i += 1

plt.show()




