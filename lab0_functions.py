import numpy as np
from numpy import array
from numpy import nan
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)


class State:
    time_horizon = None     # tba
    start_state = None
    target_state = None     # tba

    def __init__(self, index):
        self.index = index
        self.neighbours = []
        self.value = np.zeros(self.time_horizon)
        self.action = np.zeros(self.time_horizon, dtype=np.uint8)   # local index of best neighbour to go to at time t

    def __str__(self):
        return str(self.index)


class State2:
    value_tolerance = 1e-3
    start_state = None
    target_state = None

    def __init__(self, index):
        self.index = index
        self.neighbours = []
        self.value = 0
        self.next_value = 0
        self.action = 0
        self.next_action = 0

    def __str__(self):
        return 'state: ' + str(self.index)

    def update_value(self):
        no_change = np.abs(self.value - self.next_value) <= self.value_tolerance        # check convergence
        self.value = self.next_value                                    # update value
        return no_change

    def update_action(self):
        no_change = self.action == self.next_action     # check convergence
        self.action = self.next_action                  # update policy
        return no_change


def get_state_values(states, t, n, m):
    vm = np.full((n, m), nan)   # vm : value maze
    for state in states:
        vm[state.index] = state.value[t]
    return vm


def get_state_values2(states, n, m):
    vm = np.full((n, m), nan)   # vm : value maze
    for state in states:
        vm[state.index] = state.value
    return vm


def set_state_values(states):
    for state in states:
        state.value = np.random.uniform(0, 100)


def plot_arrow(s, s_next):
    x = s.index[1]
    dx = s_next.index[1] - x
    y = -s.index[0]
    dy = -s_next.index[0] - y
    plt.arrow(x, y, dx, dy, fc='k', ec='k', head_width=0.2, head_length=0.4, length_includes_head=True)
    plt.gca().set_aspect('equal')


def show_state_policies(states, t, axis):
    # plt.subplot(n, m, i) must come before calling this function
    axis.clear()
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    for state in states:
        other_state = state.neighbours[state.action[t]]
        plot_arrow(state, other_state)


def show_state_policies2(states, axis):
    # plt.subplot(n, m, i) must come before calling this function
    axis.clear()
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    for state in states:
        other_state = state.neighbours[state.action]
        plot_arrow(state, other_state)


def generate_maze():
    n_rows = 6
    n_cols = 7
    maze = np.full((n_rows, n_cols), True)
    maze[0:3, 2] = False
    maze[4, 1:6] = False
    return maze


def generate_maze_states(maze, t_horizon, start, target):
    (n_rows, n_cols) = maze.shape
    State.time_horizon = t_horizon
    states = []
    for row in range(n_rows):
        for col in range(n_cols):
            if maze[row, col]:
                state = State(index=(row, col))
                states.append(state)

                if row == start[0] and col == start[1]:
                    State.start_state = state

                elif row == target[0] and col == target[1]:
                    State.target_state = state

    for state in states:
        state.neighbours.append(state)
        for other_state in states:
            if np.abs(other_state.index[0] - state.index[0]) == 1 and np.abs(
                    other_state.index[1] - state.index[1]) == 0:
                state.neighbours.append(other_state)  # same row neighbours
            elif np.abs(other_state.index[0] - state.index[0]) == 0 and np.abs(
                    other_state.index[1] - state.index[1]) == 1:
                state.neighbours.append(other_state)  # same column neighbours
    return states


def generate_maze_states2(maze, start, target):
    (n_rows, n_cols) = maze.shape
    states = []
    for row in range(n_rows):
        for col in range(n_cols):
            if maze[row, col]:
                state = State2(index=(row, col))
                states.append(state)

                if row == start[0] and col == start[1]:
                    State2.start_state = state

                elif row == target[0] and col == target[1]:
                    State2.target_state = state

    for state in states:
        state.neighbours.append(state)
        for other_state in states:
            if np.abs(other_state.index[0] - state.index[0]) == 1 and np.abs(
                    other_state.index[1] - state.index[1]) == 0:
                state.neighbours.append(other_state)  # same row neighbours
            elif np.abs(other_state.index[0] - state.index[0]) == 0 and np.abs(
                    other_state.index[1] - state.index[1]) == 1:
                state.neighbours.append(other_state)  # same column neighbours
    return states


def get_transition_probability(states):
    n_states = len(states)
    transition_probability = np.zeros((n_states, n_states))
    for state in states:
        n_nes = len(state.neighbours)
        for ne in state.neighbours:
            transition_probability[states.index(state), states.index(ne)] = 1 / n_nes  # uniform
    return transition_probability


def generate_transition_rewards(states, target, reward_staying, reward_moving, reward_target):
    n_states = len(states)

    transition_reward = np.full((n_states, n_states), reward_moving)  # general cost of changing states

    for i in range(n_states):
        transition_reward[i, i] = reward_staying     # cost for staying at state (not target)

    nes = target.neighbours
    for ne in nes:
        transition_reward[states.index(ne), states.index(target)] = reward_target # going to / staying at the target
    return transition_reward


def backward_induction(maze, states, t_horizon, rewards, pause_time):
    fig = plt.figure()
    fig.suptitle('Backwards Induction')

    (n_rows, n_cols) = maze.shape
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    " backwards induction "
    for t in range(t_horizon - 1, -1, -1):
        for s in states:
            possible_actions = [states.index(ne) for ne in s.neighbours]  # index of possible next states
            possible_rewards = rewards[states.index(s), possible_actions]  # reward for going to the next state

            if t == t_horizon - 1:
                pass
            else:
                possible_rewards = possible_rewards + [ne.value[t + 1] for ne in s.neighbours]  # next state reward

            s.action[t] = np.argmax(possible_rewards)
            s.value[t] = possible_rewards[s.action[t]]

        values = get_state_values(states, t, n_rows, n_cols)
        ax1.matshow(values)
        ax1.set_xlabel('State values')
        show_state_policies(states, t, ax2)
        ax2.set_xlabel('State policies')
        plt.axis([-1, n_cols, -n_rows, 1])
        plt.pause(pause_time)

    return values


def plot_shortest_path(start, t_horizon, maze, pause_time):
    (n_rows, n_cols) = maze.shape
    fig = plt.figure()
    fig.suptitle('Shortest path from ' + str(start))
    axis = plt.subplot(1, 1, 1)
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    s_now = start
    for t in range(t_horizon):
        s_next = s_now.neighbours[s_now.action[t]]
        plot_arrow(s_now, s_next)
        s_now = s_next
        plt.axis([-1, n_cols, -n_rows, 1])
        plt.pause(pause_time)


def value_iteration(maze, states, rewards, gamma, pause_time):
    fig = plt.figure()
    fig.suptitle('Value Iteration')

    (n_rows, n_cols) = maze.shape
    n_states = len(states)
    value_convergence = np.full(n_states, False)
    i = 0
    ax1 = plt.subplot(2, 1, 1)
    while not all(value_convergence):
        print('i={:.0f}'.format(i))
        " VALUE ITERATION "
        for state in states:
            value_elements = array([(rewards[states.index(state), states.index(ne)] +
                                     gamma * ne.value) for ne in state.neighbours])
            state.next_value = np.max(value_elements)
        for state in states:
            value_convergence[states.index(state)] = state.update_value()
        if all(value_convergence):
            print('\tvalues: converged')
        else:
            print('\tvalues: not converged')

        values = get_state_values2(states, n_rows, n_cols)
        ax1.matshow(values)
        ax1.set_xlabel('State values')
        plt.pause(pause_time)

        i += 1

    for state in states:
        value_elements = [rewards[states.index(state), states.index(ne)] +
                          gamma * ne.value for ne in state.neighbours]
        state.action = np.argmax(array([value_elements]))

    ax2 = plt.subplot(2, 1, 2)
    show_state_policies2(states, ax2)
    ax2.set_xlabel('State policies')
    plt.gca().set_aspect('equal')
    plt.axis([-1, n_cols, -n_rows, 1])
    plt.pause(pause_time)

    return values


def policy_value_iteration(maze, states, rewards, gamma, pause_time):
    (n_rows, n_cols) = maze.shape
    n_states = len(states)
    action_convergence = np.full(n_states, False)
    value_convergence = np.full(n_states, False)
    i = 0
    fig = plt.figure()
    fig.suptitle('Policy and Value Iteration')
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    while not (all(action_convergence) and all(value_convergence)):
        print('i={:.0f}'.format(i))
        " VALUE UPDATE "
        for state in states:
            next_state = state.neighbours[state.action]     # deterministic
            state.next_value = rewards[states.index(state), states.index(next_state)] + gamma * next_state.value
        for state in states:
            value_convergence[states.index(state)] = state.update_value()
        if all(value_convergence):
            print('\tvalues: converged')
        else:
            print('\tvalues: not converged')
        " POLICY UPDATE "
        for state in states:
            value_elements = [rewards[states.index(state), states.index(ne)] +
                              gamma*ne.value for ne in state.neighbours]
            state.next_action = np.argmax(array([value_elements]))
        for state in states:
            action_convergence[states.index(state)] = state.update_action()
        if all(action_convergence):
            print('\tpolicies: converged')
        else:
            print('\tpolicies: not converged')

        values = get_state_values2(states, n_rows, n_cols)
        # print(values)

        ax1.matshow(values)
        ax1.set_xlabel('State values')
        show_state_policies2(states, ax2)
        ax2.set_xlabel('State policies')
        plt.gca().set_aspect('equal')
        plt.axis([-1, n_cols, -n_rows, 1])
        plt.pause(pause_time)
        i += 1

    return values

