from lab1_3_classes import *
from lab1_3_functions import *
import matplotlib.pyplot as plt
from numpy import array

grid_size = (4, 4)
robber_position = (0, 0)
police_position = (3, 3)
bank_position = (1, 1)

grid_size = (4, 4)
robber_position = (0, 0)
police_position = (2, 2)
bank_position = (1, 1)

states = StateSpace(grid_size=grid_size,
                    r=robber_position,
                    p=police_position,
                    b=bank_position)
rewards = Reward()

q_function = QFunction(states=states)

epsilon = 0.01

# n_iterations = 10000
# game_times = list(np.int_(n_iterations * array([0.25, 0.5, 0.8])))
# plot_games = False
#
# acc_rew = sarsa_algorithm(states=states, rewards=rewards, epsilon=epsilon, n=n_iterations, plot=plot_games, game_times=game_times)
# if plot_games:
#     game_fig = plt.figure(num=9)
#     game_ax = game_fig.add_subplot(111)
#     game_fig.suptitle('Policy after {:.0f} iterations'.format(n_iterations))
#     run_game(states=states, ax=game_ax)
#
# fig = plt.figure(num=8)
# ax = fig.add_subplot(111)
# fig.suptitle('Accumulated Reward')
# ax.plot(acc_rew[:, 0], acc_rew[:, 1])


epsilons = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]    # greedy
epsilons = [0.1]
n_iterations = 10000000
n_iterations = 5000
eis, t = evolution_of_initial_state_sarsa(states=states, rewards=rewards, n_sarsa=n_iterations, epsilons=epsilons, n_checkpoints=1000)
print(t.shape)
print(eis.shape)
fig = plt.figure(num=7)
ax = fig.add_subplot(111)
fig.suptitle('Value of Initial State')
plt.plot(t, eis)
ax.legend([str(epsilon) for epsilon in epsilons])

plt.show()

#'''