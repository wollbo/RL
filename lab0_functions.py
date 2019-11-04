import numpy as np
from numpy import array
from numpy import nan
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)


class State:
    value_tolerance = 1e-10
    target_state = None         # tba

    def __init__(self, index, in_maze=True):
        self.index = index
        self.neighbours = []
        self.value = 0
        self.next_value = 0
        self.policy = 0         # index of neighbour to go to!
        self.next_policy = 0
        self.in_maze = in_maze

    def __str__(self):
        return 'state: ' + str(self.index)

    def update_value(self):
        no_change = np.abs(self.value - self.next_value) <= self.value_tolerance        # check convergence
        self.value = self.next_value                                    # update value
        return no_change

    def update_policy(self):
        no_change = self.policy == self.next_policy     # check convergence
        self.policy = self.next_policy                  # update policy
        return no_change


def get_state_values(states, n, m):
    vm = np.full((n, m), nan)   # vm : value maze
    for state in states:
        if state.value != 0 and state.in_maze:
            vm[state.index] = state.value
    return vm


def set_state_values(states):
    for state in states:
        state.value = np.random.uniform(0, 100)


def show_state_policies(states, axis):
    # plt.subplot(n, m, i) must come before calling this function
    axis.clear()
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    for state in states:
        other_state = state.neighbours[state.policy]
        x = state.index[1]
        dx = other_state.index[1] - x
        y = -state.index[0]
        dy = -other_state.index[0] - y
        plt.arrow(x, y, dx, dy, fc='k', ec='k', head_width=0.25, head_length=0.5, length_includes_head=True)


def generate_maze_states(maze):
    (n_rows, n_cols) = maze.shape
    states = []
    for row in range(n_rows):
        for col in range(n_cols):
            if maze[row, col]:
                state = State(index=(row, col))
                states.append(state)

                if row == 5 and col == 5:
                    State.target_state = state

    for state in states:
        state.neighbours.append(state)
        for other_state in states:
            if np.abs(other_state.index[0] - state.index[0]) == 1 and np.abs(
                    other_state.index[1] - state.index[1]) == 0:
                state.neighbours.append(other_state)  # same row
            elif np.abs(other_state.index[0] - state.index[0]) == 0 and np.abs(
                    other_state.index[1] - state.index[1]) == 1:
                state.neighbours.append(other_state)  # same column
    return states


def get_transition_probability(states):
    n_states = len(states)
    transition_probability = np.zeros((n_states, n_states))
    for state in states:
        n_nes = len(state.neighbours)
        for ne in state.neighbours:
            transition_probability[states.index(state), states.index(ne)] = 1 / n_nes  # uniform
    return transition_probability

def get_transition_reward(states):
    n_states = len(states)
    transition_reward = np.zeros((n_states, n_states))
    target = State.target_state
    nes = target.neighbours
    for ne in nes:
        transition_reward[states.index(ne), states.index(target)] = 100   # reward for going to / staying at the target!
    return transition_reward

def policy_value_iteration(maze, states, transition_probability, transition_reward, gamma, pause_time):
    (n_rows, n_cols) = maze.shape
    n_states = len(states)
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
        # print(values)

        ax1.matshow(np.log(values))
        ax1.set_xlabel('State values')
        show_state_policies(states, ax2)
        ax2.set_xlabel('State policies')
        plt.gca().set_aspect('equal')
        plt.axis([-1, n_cols, -n_rows, 1])
        plt.pause(pause_time)
        i += 1

    # plt.show()

def value_iteration(maze, states, transition_probability, transition_reward, gamma, pause_time):
    (n_rows, n_cols) = maze.shape
    n_states = len(states)
    value_convergence = np.full(n_states, False)
    i = 0
    ax1 = plt.subplot(2, 1, 1)
    while not all(value_convergence):
        print('i={:.0f}'.format(i))
        " VALUE ITERATION UPDATE "
        for state in states:
            value_elements = array([transition_probability[states.index(state), states.index(ne)] *
                                    (transition_reward[states.index(state), states.index(ne)] +
                                     gamma * ne.value) for ne in state.neighbours])
            state.next_value = np.max(value_elements)
        for state in states:
            value_convergence[states.index(state)] = state.update_value()
        if all(value_convergence):
            print('\tvalues: converged')
        else:
            print('\tvalues: not converged')

        values = get_state_values(states, n_rows, n_cols)
        # print(values)

        ax1.matshow(np.log(values))
        ax1.set_xlabel('State values')
        plt.pause(pause_time)
        i += 1

    for state in states:
        value_elements = [transition_reward[states.index(state), states.index(ne)] +
                          gamma * ne.value for ne in state.neighbours]
        state.policy = np.argmax(array([value_elements]))

    ax2 = plt.subplot(2, 1, 2)

    show_state_policies(states, ax2)
    ax2.set_xlabel('State policies')
    plt.gca().set_aspect('equal')
    plt.axis([-1, n_cols, -n_rows, 1])
    plt.pause(pause_time)

    # plt.show()
