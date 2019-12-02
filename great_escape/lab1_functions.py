import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy.random import choice
from scipy.stats import geom

# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

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
    if states.finite_horizon:
        print('Performing Backward Induction')
    else:
        print('Can not perform value iteration on these states')
        return None
    time_horizon = states.time_horizon
    for t in range(time_horizon - 1, -1, -1):
        print('\t induction at t = {:.0f}'.format(t))
        for state in states:
            player = state.player

            state_reward = array(rewards(state))    # reward for current state

            if t == time_horizon - 1:  # first (last) step
                player.action[t] = player.neighbours.index(player)  # stand still in last step, no possible action
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
    while not states.stopping_condition(i=i):
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


def probability_of_exiting(states, n_tests):
    score = []
    if states.finite_horizon:
        game_horizon = states.time_horizon
    elif states.infinite_horizon_discounted:
        time_horizon_distribution = geom(p=1 / states.lifetime_mean)

    for _ in range(n_tests):

        if states.infinite_horizon_discounted:
            game_horizon = time_horizon_distribution.rvs()
        state = states.initial_state
        t = 0
        while True:
            if state.losing():
                score.append(False)
                break
            elif state.winning():
                score.append(True)
                break
            elif t == game_horizon - 1:
                score.append(False)
                break

            if states.finite_horizon:
                action = state.player.action[t]
            else:
                action = state.player.action

            state = choice(states.possible_next_states(state, action))
            t += 1

    return sum(score)/n_tests


def run_finite_game_show_values(states, pause_time=0.6):
    if states.finite_horizon:
        game_horizon = states.time_horizon
    elif states.infinite_horizon_discounted:
        time_horizon_distribution = geom(p=1 / states.lifetime_mean)
        game_horizon = time_horizon_distribution.rvs()

    expected_reward = np.full(game_horizon, np.nan)
    current_state = states.initial_state    # starting position
    t = 0

    fig = plt.figure(figsize=(6, 4))
    grid = plt.GridSpec(2, 1, wspace=0.4, hspace=0.3)
    game_ax = fig.add_subplot(grid[0, 0])
    game_ax.matshow(states.maze, cmap=plt.cm.gray)
    exp_ax = fig.add_subplot(grid[1, 0])
    exp_ax.set_xlabel('Expected reward', labelpad=0)
    exp_ax.set_xticks(range(game_horizon))
    fig.suptitle('t={:.0f}'.format(t))
    (p_2, p_1) = current_state.player.position
    (m_2, m_1) = current_state.minotaur.position
    player_marker, = game_ax.plot(p_1, p_2, color='blue', marker='$P$', markersize=9)
    minotaur_marker, = game_ax.plot(m_1, m_2, color='red', marker='$M$', markersize=9)
    plt.pause(pause_time)

    while True:
        if states.finite_horizon:
            expected_reward[t] = current_state.value[t]
        elif states.infinite_horizon_discounted:
            expected_reward[t] = current_state.value

        print('At time {:.0f} of {:.0f}'.format(t, game_horizon))
        print(current_state)
        fig.suptitle('t={:.0f}, T = {:.0f}'.format(t, game_horizon))

        (p_4, p_3) = (p_2, p_1)
        (m_4, m_3) = (m_2, m_1)

        (p_2, p_1) = current_state.player.position
        (m_2, m_1) = current_state.minotaur.position
        if (p_2, p_1) != (p_4, p_3):
            game_ax.arrow(p_3, p_4, p_1 - p_3, p_2 - p_4, fc='blue', ec='blue', head_width=0.2, head_length=0.4, length_includes_head=True)
        else:
            game_ax.plot(p_1, p_2, marker='o', color='blue')

        game_ax.arrow(m_3, m_4, m_1 - m_3, m_2 - m_4, fc='red', ec='red', head_width=0.2, head_length=0.4, length_includes_head=True)

        player_marker.set_data(p_1, p_2)
        minotaur_marker.set_data(m_1, m_2)
        exp_ax.clear()
        exp_ax.stem(expected_reward, use_line_collection=True)
        exp_ax.set_xticks(range(game_horizon))
        plt.pause(pause_time)

        if current_state.losing():
            print('\nLosing by death\n')
            break
        elif current_state.winning():
            print('\nWinning\n')
            break
        elif t == game_horizon - 1:
            print('\nLosing by Time Out\n')
            break

        if states.finite_horizon:
            action = current_state.player.action[t]
        elif states.infinite_horizon_discounted:
            action = current_state.player.action

        pns = states.possible_next_states(current_state, action)
        next_state = choice(pns)    # random minotaur movement
        current_state = next_state
        t += 1


def run_finite_game(states, pause_time=0.3):
    if states.finite_horizon:
        game_horizon = states.time_horizon
    elif states.infinite_horizon_discounted:
        time_horizon_distribution = geom(p=1 / states.lifetime_mean)
        game_horizon = time_horizon_distribution.rvs()

    current_state = states.initial_state    # starting position
    t = 0

    fig = plt.figure(num=13)
    fig.clear()
    game_ax = fig.add_subplot(111)
    game_ax.matshow(states.maze, cmap=plt.cm.gray)
    (p_2, p_1) = current_state.player.position
    (m_2, m_1) = current_state.minotaur.position
    player_marker, = game_ax.plot(p_1, p_2, color='blue', marker='$P$', markersize=16)
    minotaur_marker, = game_ax.plot(m_1, m_2, color='red', marker='$M$', markersize=16)
    plt.pause(pause_time)

    while True:
        print('At time {:.0f} of {:.0f}'.format(t, game_horizon))
        print(current_state)
        #fig.suptitle('t={:.0f}, T = {:.0f}'.format(t, game_horizon))

        (p_4, p_3) = (p_2, p_1)
        (m_4, m_3) = (m_2, m_1)

        (p_2, p_1) = current_state.player.position
        (m_2, m_1) = current_state.minotaur.position

        if (p_2, p_1) != (p_4, p_3):
            game_ax.arrow(p_3, p_4, p_1 - p_3, p_2 - p_4, fc='blue', ec='blue', head_width=0.2, head_length=0.4, length_includes_head=True)
        else:
            game_ax.plot(p_1, p_2, marker='o', color='blue')

        game_ax.arrow(m_3, m_4, m_1 - m_3, m_2 - m_4, fc='red', ec='red', head_width=0.2, head_length=0.4, length_includes_head=True)

        player_marker.set_data(p_1, p_2)
        minotaur_marker.set_data(m_1, m_2)
        plt.pause(pause_time)

        if current_state.losing():
            print('\nLosing by death\n')
            break
        elif current_state.winning():
            print('\nWinning\n')
            break
        elif t == game_horizon - 1:
            print('\nLosing by Time Out\n')
            break

        if states.finite_horizon:
            action = current_state.player.action[t]
        elif states.infinite_horizon_discounted:
            action = current_state.player.action

        pns = states.possible_next_states(current_state, action)
        next_state = choice(pns)    # random minotaur movement
        current_state = next_state
        t += 1


def show_policies(states, maze, minotaur_position, t=0):

    fig = plt.figure(num=16)
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    #if states.finite_horizon:
        #fig.suptitle('Optimal Policies at t={:.0f}, T = {:.0f}'.format(t, states.time_horizon))
        #ax.set_xlabel('$\pi (s_t)$', labelpad=17)
    #elif states.infinite_horizon_discounted:
        #fig.suptitle('Optimal Policies, $\lambda =  {:.3}$'.format(states.discount_factor))
        #ax.set_xlabel('$\pi (s)$', labelpad=17)

    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

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
    ax.plot(m_x, m_y, color='r', marker='$M$', markersize=16)


def show_values(states, maze, minotaur_position, t=0):

    fig = plt.figure(num=15)
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    #if states.finite_horizon:
        #fig.suptitle('Expected Rewards at t={:.0f}, T = {:.0f}'.format(t, states.time_horizon))
        #ax.set_xlabel('$V_T^\pi (s)$', labelpad=17)
    #elif states.infinite_horizon_discounted:
        #fig.suptitle('Expected Rewards, $\lambda$ = {:.2f}'.format(states.discount_factor))
        #ax.set_xlabel('$V_\lambda^\pi (s)$', labelpad=12)
    #ax.set_yticklabels([])
    #ax.set_yticks([])
    #ax.set_xticklabels([])
    #ax.set_xticks([])

    minotaur = [s for s in states.minotaur_states if s.position == minotaur_position][0]
    values = np.full(maze.shape, np.nan)

    for state in states.subset_where(minotaur_id=minotaur.id):
        if states.finite_horizon:
            values[state.player.position] = state.value[t]
        elif states.infinite_horizon_discounted:
            values[state.player.position] = state.value

    pos = ax.matshow(values, vmin=0, vmax=1)        # cmap=plt.cm.autumn

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            if not np.isnan(values[r, c]):
                val = values[r, c]
                if val == 1.0 or val == 0.0:
                    ax.plot(c, r, marker='${:.0f}$'.format(val), color='b', markersize=11)
                else:
                    mark = '${:.2f}$'.format(val)
                    ax.plot(c, r, marker=mark, color='b', markersize=28)



    (m_y, m_x) = minotaur.position
    ax.plot(m_x, m_y, color='r', marker='$M$', markersize=16)

    fig.colorbar(pos, ax=ax, fraction=0.04)


