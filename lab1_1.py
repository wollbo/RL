from lab1_functions import *
from lab1_classes import *
from numpy.random import choice


maze = generate_maze()
time_horizon = 20


states = StateSpace(maze=maze, time_horizon=time_horizon, exit=(6, 5))
rewards = Reward(reward_eaten=-100, reward_exiting=100)
probabilities = None

backward_induction(states=states, rewards=rewards)


fig, ax = plt.subplots(1)
for _ in range(20):

    state = states.initial_state
    for t in range(time_horizon):
        print('At time {:.0f}'.format(t))
        print(state)
        print('Current expected reward: {:.0f}'.format(state.value[t]))

        p_y = state.player.position[0]
        p_x = state.player.position[1]
        m_y = state.minotaur.position[0]
        m_x = state.minotaur.position[1]

        plt.cla()
        ax.matshow(maze, cmap=plt.cm.gray)
        ax.plot(p_x, p_y, 'bo')
        ax.plot(m_x, m_y, 'r+')

        action = state.player.action[t]
        pns = states.possible_next_states(state, action)
        next_state = choice(pns)

        if state.player.position == next_state.player.position:
            plt.xlabel('Player stood still')
            plt.pause(2)
        else:
            plt.xlabel('Player moved')
            plt.pause(0.2)

        state = next_state



plt.show()
