import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy.random import choice

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


def generate_maze():
    n_rows = 7
    n_cols = 8
    maze = np.full((n_rows, n_cols), True)
    maze[0:4, 2] = False
    maze[5, 1:7] = False
    maze[6, 4] = False
    maze[1:4, 5] = False
    maze[2, 5:8] = False

    return maze


def backward_induction(states, rewards):
    time_horizon = states.time_horizon
    for t in range(time_horizon - 1, -1, -1):
        print('Backwards induction at t = {:.0f}'.format(t))
        for state in states:
            player = state.player

            state_reward = array(rewards(state))    # reward for current state

            if t == time_horizon - 1:  # first (last) step
                player.action[t] = player.neighbours.index(player)  # stand still
                state.value[t] = state_reward

            else:  # other steps
                possible_actions = np.r_[:len(player.neighbours)]  # action is local index of neighbour
                possible_next_rewards = np.zeros(len(possible_actions))

                for action in possible_actions:
                    pns = states.possible_next_states(state, action)
                    p = 1 / len(pns)  # uniform probability
                    possible_next_rewards[action] = np.sum(p * ne.value[t + 1] for ne in pns)  # next state reward

                possible_rewards = array([state_reward + possible_next_rewards[a] for a in possible_actions])
                player.action[t] = np.argmax(array(possible_rewards))
                state.value[t] = possible_rewards[player.action[t]]


def value_iteration(states, rewards, plot=False):
    if states.infinite_horizon_discounted:
        print('Performing Value Iteration')
    else:
        print('Can not perform value iteration on these states')
        return None
    gamma = states.discount_factor
    i = 0
    while not states.stopping_condition():
        print('i={:.0f}'.format(i))
        for state in states:
            player = state.player

            possible_actions = np.r_[:len(player.neighbours)]  # action is local index of neighbour
            possible_next_rewards = np.zeros(len(possible_actions))

            for action in possible_actions:
                pns = states.possible_next_states(state, action)
                p = 1 / len(pns)        # uniform probability
                possible_next_rewards[action] = rewards(state) + gamma * np.sum(p * ne.value for ne in pns)

            state.next_value = np.max(possible_next_rewards)

        states.update_values()

        if plot:
            show_values(states=states, maze=states.maze, minotaur_position=states.exit)
            plt.pause(0.3)

        if states.stopping_condition():
            print('\tvalues: converged')
        else:
            print('\tvalues: not converged')

        i += 1

    for state in states:
        player = state.player
        possible_actions = np.r_[:len(player.neighbours)]  # action is local index of neighbour
        possible_next_rewards = np.zeros(len(possible_actions))
        for action in possible_actions:
            pns = states.possible_next_states(state, action)
            p = 1 / len(pns)  # uniform probability
            possible_next_rewards[action] = rewards(state) + gamma * np.sum(p * ne.value for ne in pns)

        player.action = np.argmax(possible_next_rewards)

    return i


def run_game(maze, states, rewards, pause_time=0.6):

    fig = plt.figure(figsize=(10, 4))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

    expected_reward = np.full(states.time_horizon, np.nan)
    received_reward = np.full(states.time_horizon, np.nan)

    current_state = states.initial_state    # starting position
    for t in range(states.time_horizon):

        expected_reward[t] = current_state.value[t]
        received_reward[t] = rewards(current_state)

        print('At time {:.0f}'.format(t))
        print(current_state)

        (p_2, p_1) = current_state.player.position
        (m_2, m_1) = current_state.minotaur.position

        game_ax = fig.add_subplot(grid[:, 0])
        exp_ax = fig.add_subplot(grid[0, 1])
        rew_ax = fig.add_subplot(grid[1, 1])
        exp_ax.stem(expected_reward, use_line_collection=True)
        exp_ax.set_xticks(range(states.time_horizon))
        exp_ax.set_yticks(range(6))
        rew_ax.stem(np.cumsum(received_reward), use_line_collection=True)
        rew_ax.set_xticks(range(states.time_horizon))
        rew_ax.set_yticks(range(6))
        game_ax.matshow(maze, cmap=plt.cm.gray)
        game_ax.plot(p_1, p_2, 'bo')
        game_ax.plot(m_1, m_2, 'rx')
        plt.pause(pause_time)
        plt.clf()

        if states.finite_horizon:
            action = current_state.player.action[t]
        elif states.infinite_horizon_discounted:
            action = current_state.player.action

        pns = states.possible_next_states(current_state, action)
        next_state = choice(pns)    # random minotaur movement
        current_state = next_state



def show_policies(states, maze, minotaur_position, t=0):
    fig = plt.figure(num=16)
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    if states.finite_horizon:
        fig.suptitle('Optimal Policies at t={:.0f}, T = {:.0f}'.format(t, states.time_horizon))
        ax.set_xlabel('$\pi (s_t)$', labelpad=17)
    elif states.infinite_horizon_discounted:
        fig.suptitle('Optimal Policies, $\lambda =  {:.2}$'.format(states.discount_factor))
        ax.set_xlabel('$\pi (s)$', labelpad=17)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.matshow(maze, cmap=plt.cm.gray)

    minotaur = [s for s in states.minotaur_states if s.position == minotaur_position][0]

    for state in states.subset_where(minotaur_id=minotaur.id):
        if states.finite_horizon:
            action = state.player.action[t]
        elif states.infinite_horizon_discounted:
            action = state.player.action

        (y1, x1) = state.player.position
        (y2, x2) = state.player.neighbours[action].position

        if (y1, x1) == (y2, x2):
            plt.plot(x1, y1, 'ko')
        else:
            plt.arrow(x1, y1, x2-x1, y2-y1, fc='k', ec='k', head_width=0.2, head_length=0.4, length_includes_head=True)

    (m_y, m_x) = minotaur.position
    ax.plot(m_x, m_y, 'rx')


def show_values(states, maze, minotaur_position, t=0):
    fig = plt.figure(num=15)
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    if states.finite_horizon:
        fig.suptitle('Expected Rewards at t={:.0f}, T = {:.0f}'.format(t, states.time_horizon))
        ax.set_xlabel('$V_T^\pi (s)$', labelpad=17)
    elif states.infinite_horizon_discounted:
        fig.suptitle('Expected Rewards, $\lambda$ = {:.2f}'.format(states.discount_factor))
        ax.set_xlabel('$V_\lambda^\pi (s)$', labelpad=17)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    minotaur = [s for s in states.minotaur_states if s.position == minotaur_position][0]
    values = np.full(maze.shape, np.nan)

    for state in states.subset_where(minotaur_id=minotaur.id):
        if states.finite_horizon:
            values[state.player.position] = state.value[t]
        elif states.infinite_horizon_discounted:
            values[state.player.position] = state.value

    pos = ax.matshow(values)        # cmap=plt.cm.autumn
    (m_y, m_x) = minotaur.position
    ax.plot(m_x, m_y, 'rx')

    fig.colorbar(pos, ax=ax, fraction=0.04)


