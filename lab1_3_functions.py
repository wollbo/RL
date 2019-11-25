from lab1_3_classes import *
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import matplotlib.cm as matplotlib_colormap


def run_game(states, ax):
    max_game_time = 50
    ax.clear()
    grid = np.ones(states.grid_size)
    grid[states.bank_position] = 0
    colormap = matplotlib_colormap.get_cmap('RdBu')
    ax.matshow(grid, cmap=colormap)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    state = states.initial_state
    (r2, r1) = state.robber.position
    (p2, p1) = state.police.position
    robber_mark, = ax.plot(r1, r2, color=colormap(0.15), marker='$R$', markersize=16)     # create mark for robber
    police_mark, = ax.plot(p1, p2, color=colormap(0.85), marker='$P$', markersize=16)     # create mark for police
    plt.pause(0.5)                          # pause and draw
    i = 0
    while not state.robber_caught() and i < max_game_time:
        i += 1
        state = choice(states.possible_next_states(state=state, action=state.robber.action))
        (r2, r1) = state.robber.position
        (p2, p1) = state.police.position
        robber_mark.set_data(r1, r2)
        police_mark.set_data(p1, p2)
        plt.pause(0.2)  # pause and update (draw) positions


def sarsa_algorithm(states, rewards, epsilon=0.2, n=10000000, n_checkpoints=500, plot=False, game_times=[]):
    if plot:
        game_fig = plt.figure(num=10)
        game_ax = game_fig.add_subplot(111)
    states.reset_actions()
    q_function = QFunction(states=states)
    state = states.initial_state
    # accumulated_reward_array = np.full((n_checkpoints, 2), np.nan)
    # accumulated_reward = 0
    value_initial_state = np.full((n_checkpoints, 2), np.nan)
    is_id = states.initial_state.id
    is_robber = states.initial_state.robber
    for t in range(n):
        reward = rewards(state=state)
        # accumulated_reward += reward
        action = state.robber.select_action(epsilon=epsilon)
        possible_next_states = states.possible_next_states(state=state, action=action)
        next_state = choice(possible_next_states)
        next_action = next_state.robber.action

        q_function.update_sarsa(state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                next_action=next_action)

        if t % round(n/n_checkpoints) == 0:
            print(t / n)
            value_initial_state[round(t*n_checkpoints/n), 0] = t
            value_initial_state[round(t*n_checkpoints/n), 1] = q_function.q_values[is_id][is_robber.action].value
            # accumulated_reward_array[round(t*n_checkpoints/n), 0] = t
            # accumulated_reward_array[round(t*n_checkpoints/n), 1] = accumulated_reward

        if t in game_times and plot:
            game_fig.suptitle('Policy after {:.0f} iterations of {:.0f}'.format(t, n))
            run_game(states=states, ax=game_ax)

        state = next_state

    # return accumulated_reward_array
    return value_initial_state


def evolution_of_initial_state(states, rewards, n_sarsa, epsilons, n_checkpoints=500):
    values_initial_state = np.full((n_checkpoints, len(epsilons)), np.nan)
    for n in range(len(epsilons)):
        print('\t\t\t ' + str(n) + ' ' + str(len(epsilons)))
        v_i_s = sarsa_algorithm(states=states,
                                rewards=rewards,
                                epsilon=epsilons[n],
                                n=n_sarsa, n_checkpoints=n_checkpoints)
        np.savetxt(X=v_i_s, fmt=['%10.0f', '%10.2e'],
                   fname='eis_e={:.3f}.txt'.format(epsilons[n]))
                   #fname='eis_e=' + str(epsilons[n]) + '.txt')

        values_initial_state[:, n] = v_i_s[:, 1]
    t = v_i_s[:, 0]

    return values_initial_state, t


def learning_curves(states, rewards, n_sarsa, n_averaging, epsilons, n_checkpoints=500):
    averaged_accumulated_rewards = np.full((n_checkpoints, len(epsilons)), np.nan)
    for n in range(len(epsilons)):
        accumulated_rewards = np.full((n_checkpoints, n_averaging), np.nan)
        for m in range(n_averaging):
            acc_rew = sarsa_algorithm(states=states, rewards=rewards, epsilon=epsilons[n], n=n_sarsa, n_checkpoints=n_checkpoints)
            accumulated_rewards[:, m] = acc_rew[:, 1]
        t = acc_rew[:, 0]
        averaged_accumulated_rewards[:, n] = np.mean(accumulated_rewards, axis=1)
        t_aar = np.stack((t, averaged_accumulated_rewards[:, n]), axis=1)
        np.savetxt(X=t_aar, fmt=['%10.0f', '%10.2e'],
                   fname='aar_e=' + str(epsilons[n]) + '_averages=' + str(n_averaging) + '.txt')

    return averaged_accumulated_rewards, t
