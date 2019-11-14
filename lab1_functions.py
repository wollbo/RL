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

    " backwards induction "
    for t in range(time_horizon - 1, -1, -1):
        print('Backwards induction at t = {:.0f}'.format(t))
        for state in states:
            player = state.player
            minotaur = state.minotaur

            state_reward = array(rewards.reward(player, minotaur))  # reward for current state

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


def run_game(maze, states, pause_time=0.3):
    fig, ax = plt.subplots(1)
    current_state = states.initial_state
    for t in range(states.time_horizon):
        print('At time {:.0f}'.format(t))
        print(current_state)
        print('Current expected reward: {:.0f}'.format(current_state.value[t]))

        (p_y, p_x) = current_state.player.position
        (m_y, m_x) = current_state.minotaur.position

        plt.cla()
        ax.matshow(maze, cmap=plt.cm.gray)
        ax.plot(p_x, p_y, 'bo')
        ax.plot(m_x, m_y, 'rx')
        plt.pause(pause_time)

        action = current_state.player.action[t]
        pns = states.possible_next_states(current_state, action)
        next_state = choice(pns)
        current_state = next_state


def show_policies(states, maze, t, minotaur_position):
    fig, ax = plt.subplots(1)
    ax.matshow(maze, cmap=plt.cm.gray)
    for state in states:
        if state.minotaur.position == minotaur_position:
            (y1, x1) = state.player.position
            action = state.player.action[t]
            (y2, x2) = state.player.neighbours[action].position

            if (y1, x1) == (y2, x2):
                plt.plot(x1, y1, 'ko')
            else:
                plt.arrow(x1, y1, x2-x1, y2-y1, fc='k', ec='k', head_width=0.2, head_length=0.4, length_includes_head=True)

    (m_x, m_y) = minotaur_position
    ax.plot(m_x, m_y, 'rx')


