import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

rc('font', size=SMALL_SIZE)  # controls default text sizes
rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

np.set_printoptions(precision=2)


def plot_arrow(s, s_next):
    x = s.index[1]
    dx = s_next.index[1] - x
    y = -s.index[0]
    dy = -s_next.index[0] - y
    plt.arrow(x, y, dx, dy, fc='k', ec='k', head_width=0.2, head_length=0.4, length_includes_head=True)
    plt.gca().set_aspect('equal')


# def show_state_policies(states, t, axis):
#     # plt.subplot(n, m, i) must come before calling this function
#     axis.clear()
#     axis.set_yticklabels([])
#     axis.set_xticklabels([])
#     for state in states:
#         other_state = state.neighbours[state.action[t]]
#         plot_arrow(state, other_state)


# def show_state_policies2(states, axis):
#     # plt.subplot(n, m, i) must come before calling this function
#     axis.clear()
#     axis.set_yticklabels([])
#     axis.set_xticklabels([])
#     for state in states:
#         other_state = state.neighbours[state.action]
#         plot_arrow(state, other_state)


def generate_maze():
    n_rows = 6
    n_cols = 7
    maze = np.full((n_rows, n_cols), True)
    maze[0:3, 2] = False
    maze[4, 1:6] = False
    return maze


def backward_induction(maze, states, rewards, plot, pause_time):
    (n_rows, n_cols) = maze.shape
    time_horizon = states.time_horizon

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)

    " backwards induction "
    for t in range(time_horizon - 1, -1, -1):
        for s in states:
            possible_actions = [ne.id for ne in s.neighbours]  # index of possible next states
            possible_rewards = array([rewards[s.id, action] for action in possible_actions]) # reward for going to the next state

            if t == time_horizon - 1:  # first step
                pass
            else:                   # other steps
                possible_rewards = possible_rewards + [ne.value[t + 1] for ne in s.neighbours]  # next state reward

            s.action[t] = np.argmax(array(possible_rewards))
            s.value[t] = possible_rewards[s.action[t]]

        values = states.values(t=t)
        if plot:
            fig.suptitle('Backwards Induction\nT = {:.0f}'.format(time_horizon))

            ax1.matshow(values)
            ax1.set_xlabel('State values')
            states.show_policies(t=t, axis=ax2)
            ax2.set_xlabel('State policies at t = T-{:.0f} = {:.0f}'.format(time_horizon - t, t))
            plt.axis([-1, n_cols, -n_rows, 1])
            plt.pause(pause_time)

    return values


def value_iteration(maze, states, rewards, plot, pause_time):
    gamma = states.discount
    (n_rows, n_cols) = maze.shape
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Value Iteration')

    i = 0

    while not states.stopping_condition():
        print('i={:.0f}'.format(i))
        " VALUE ITERATION "
        for state in states:
            vals = array([(rewards[state.id, ne.id] + gamma * ne.value) for ne in state.neighbours])
            state.next_value = np.max(vals)

        states.update_values()

        if states.stopping_condition():
            print('\tvalues: converged')
        else:
            print('\tvalues: not converged')

        i += 1

        if plot:
            values = states.values()
            ax1.matshow(values)
            ax1.set_xlabel('State values')
            plt.pause(pause_time)
            for state in states:
                value_elements = [rewards[state.id, ne.id] +
                                  gamma * ne.value for ne in state.neighbours]
                state.action = np.argmax(array([value_elements]))
            states.show_policies(axis=ax2)
            ax2.set_xlabel('State policies')
            plt.gca().set_aspect('equal')
            plt.axis([-1, n_cols, -n_rows, 1])
            plt.pause(pause_time)

    values = states.values()

    for state in states:
        value_elements = [rewards[state.id, ne.id] +
                          gamma * ne.value for ne in state.neighbours]
        state.action = np.argmax(array([value_elements]))

    return values, i


def policy_iteration(maze, states, rewards, plot, pause_time):
    (n_rows, n_cols) = maze.shape
    n_states = len(states)
    gamma = states.discount
    action_convergence = np.full(n_states, False)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Policy Iteration')

    i = 0

    while not all(action_convergence):
        print('i={:.0f}'.format(i))

        " POLICY EVALUATION "
        reward_array = array([rewards[s.id, s.neighbours[s.action].id] for s in states])
        probability_matrix = np.zeros((n_states, n_states))
        for state in states:
            probability_matrix[state.id, state.neighbours[state.action].id] = 1

        # find the equilibrium of expected reward under the given policy
        value_array = np.linalg.inv(np.eye(n_states) - gamma * probability_matrix) @ reward_array
        for state in states:
            state.value = value_array[state.id]

        " POLICY UPDATE "
        for state in states:
            value_elements = [rewards[state.id, ne.id] +
                              gamma * ne.value for ne in state.neighbours]
            state.next_action = np.argmax(array([value_elements]))
        for state in states:
            action_convergence[state.id] = state.update_action()
        if all(action_convergence):
            print('\tpolicies: converged')
        else:
            print('\tpolicies: not converged')

        i += 1

        if plot:
            values = states.values()
            ax1.matshow(values)
            ax1.set_xlabel('State values')
            states.show_policies(axis=ax2)
            ax2.set_xlabel('State policies')
            plt.gca().set_aspect('equal')
            plt.axis([-1, n_cols, -n_rows, 1])
            plt.pause(pause_time)

    values = states.values()
    return values, i


def plot_most_rewarding_path(states, pause_time):
    values = states.values(t=0)
    (n_rows, n_cols) = values.shape

    s0 = states.start

    t_horizon = states.time_horizon
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle('Backward Induction\nT={:.0f}'.format(t_horizon))
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


def plot_most_rewarding_policy(states, n, algorithm):
    gamma = states.discount
    values = states.values()
    (n_rows, n_cols) = values.shape
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(algorithm +
                 '\ndiscount $\lambda$ = {:.2f}, '
                 'required iterations: {:.0f}'.format(gamma, n))
    pos = ax1.matshow(values)
    ax1.set_xlabel('$V_{\lambda}^{\pi} (s)$', labelpad=17)
    fig.colorbar(pos, ax=ax1, fraction=0.04)
    states.show_policies(axis=ax2)
    ax2.set_xlabel('Most rewarding policy')
    plt.axis([-1, n_cols, -n_rows, 1])
