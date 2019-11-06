import numpy as np
from numpy import array
from numpy import nan
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

rc('font', size=SMALL_SIZE)          # controls default text sizes
rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

np.set_printoptions(precision=2)


class State:        # state for finite horizon problems
    time_horizon = None
    start = None
    target = None

    def __init__(self, index):
        self.index = index
        self.neighbours = []
        self.value = np.zeros(self.time_horizon)
        self.action = np.zeros(self.time_horizon, dtype=np.uint8)   # local index of best neighbour to go to at time t

    def __str__(self):
        return str(self.index)


class State2:       # state for infinite horizon problems
    value_tolerance = 1e-3
    start = None
    target = None

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


def generate_maze_states(maze, t_horizon, start=(None, None), target=(None, None)):
    (n_rows, n_cols) = maze.shape
    State.time_horizon = t_horizon
    states = []
    for row in range(n_rows):
        for col in range(n_cols):
            if maze[row, col]:
                state = State(index=(row, col))
                states.append(state)

                if row == start[0] and col == start[1]:
                    State.start = state

                elif row == target[0] and col == target[1]:
                    State.target = state

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


def generate_maze_states2(maze, start=(None, None), target=(None, None)):
    (n_rows, n_cols) = maze.shape
    states = []
    for row in range(n_rows):
        for col in range(n_cols):
            if maze[row, col]:
                state = State2(index=(row, col))
                states.append(state)

                if (row, col) == start:
                    State2.start = state

                elif (row, col) == target:
                    State2.target = state

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


def generate_transitions_rewards2(weights, states):
    n_states = len(states)
    transition_reward = np.zeros((n_states, n_states))

    for state in states:
        for ne in state.neighbours:
            if ne != state:
                transition_reward[states.index(ne), states.index(state)] = weights[state.index]

    return transition_reward


def generate_transition_rewards(states, target, reward_staying, reward_moving, reward_target):
    n_states = len(states)

    transition_reward = np.full((n_states, n_states), reward_moving)  # general cost of changing states

    for i in range(n_states):
        transition_reward[i, i] = reward_staying     # cost for staying at state (not target)

    nes = target.neighbours
    for ne in nes:
        transition_reward[states.index(ne), states.index(target)] = reward_target # going to / staying at the target
    return transition_reward


def backward_induction(maze, states, t_horizon, rewards, plot, pause_time):
    (n_rows, n_cols) = maze.shape

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Backwards Induction, T = {:.0f}'.format(t_horizon))


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
        if plot:
            ax1.matshow(values)
            ax1.set_xlabel('State values')
            show_state_policies(states, t, ax2)
            ax2.set_xlabel('State policies')
            plt.axis([-1, n_cols, -n_rows, 1])
            plt.pause(pause_time)

    return values


def plot_most_rewarding_path(s0, values, pause_time):
    (n_rows, n_cols) = values.shape

    t_horizon = s0.time_horizon
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle('Backward Induction, T={:.0f}'.format(t_horizon))
    ax1.set_xlabel('$V_T^\pi (s)$', labelpad=17)
    pos = ax1.matshow(values)
    fig.colorbar(pos, ax=ax1, fraction=0.04)
    ax2.set_xlabel('Most rewarding path')

    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    for t in range(t_horizon):
        s1 = s0.neighbours[s0.action[t]]
        plot_arrow(s0, s1)
        s0 = s1
        plt.axis([-1, n_cols, -n_rows, 1])
        plt.pause(pause_time)


def plot_most_rewarding_policy(states, values, gamma, n):
    (n_rows, n_cols) = values.shape
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Infinite Horizon, '
                 '$\lambda$ = {:.3f} \n'
                 'tolerance = {:.0e}, '
                 'required iterations: {:.0f}'.format(gamma,
                                                      State2.value_tolerance,
                                                      n))
    pos = ax1.matshow(values)
    ax1.set_xlabel('$V_{\lambda}^{\pi} (s)$', labelpad=17)
    fig.colorbar(pos, ax=ax1, fraction=0.04)
    show_state_policies2(states=states, axis=ax2)
    ax2.set_xlabel('Most rewarding policy')
    plt.axis([-1, n_cols, -n_rows, 1])


def value_iteration(maze, states, rewards, gamma, plot, pause_time):
    (n_rows, n_cols) = maze.shape
    n_states = len(states)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Value Iteration')

    value_convergence = np.full(n_states, False)
    i = 0
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

        if plot:
            ax1.matshow(values)
            ax1.set_xlabel('State values')
            plt.pause(pause_time)

        i += 1

    for state in states:
        value_elements = [rewards[states.index(state), states.index(ne)] +
                          gamma * ne.value for ne in state.neighbours]
        state.action = np.argmax(array([value_elements]))

    if plot:
        show_state_policies2(states, ax2)
        ax2.set_xlabel('State policies')
        plt.gca().set_aspect('equal')
        plt.axis([-1, n_cols, -n_rows, 1])
        plt.pause(pause_time)

    return values, gamma, i


def policy_value_iteration(maze, states, rewards, gamma, pause_time):
    (n_rows, n_cols) = maze.shape
    n_states = len(states)
    action_convergence = np.full(n_states, False)
    value_convergence = np.full(n_states, False)
    i = 0
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Policy and Value Iteration')

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

