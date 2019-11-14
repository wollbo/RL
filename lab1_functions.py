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
        for state in states:
            player = state.player
            minotaur = state.minotaur

            state_reward = array(rewards.reward(player, minotaur))  # reward for current state

            if t == time_horizon - 1:  # first step

                player.action[t] = player.neighbours.index(player)  # stand still
                state.value[t] = state_reward

            else:                   # other steps
                p = 1/len(minotaur.neighbours)

                possible_actions = np.r_[:len(player.neighbours)]     # action is local index of neighbour
                possible_next_rewards = np.zeros(len(possible_actions))

                for action in possible_actions:
                    pns = states.possible_next_states(state, action)
                    possible_next_rewards[action] = np.sum(p * ne.value[t + 1] for ne in pns)  # next state reward

                possible_rewards = array([state_reward + possible_next_rewards[a] for a in possible_actions])
                player.action[t] = np.argmax(array(possible_rewards))
                state.value[t] = possible_rewards[player.action[t]]



