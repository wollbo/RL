from lab1_3_classes import *
from lab1_3_functions import *
from numpy.random import choice

grid_size = (4, 4)
robber_position = (0, 0)
police_position = (3, 3)
bank_position = (1, 1)

states = StateSpace(grid_size=grid_size,
                    r=robber_position,
                    p=police_position,
                    b=bank_position)
rewards = Reward()
n = 1000000
n_checkpoints = 500
game_times = np.int_(n*np.array([0.5, 0.7, 0.9]))
# vis = q_learning_algorithm(states=states, rewards=rewards, n=n, n_checkpoints=n_checkpoints, plot=True, game_times=game_times)
vis = q_learning_algorithm(states=states, rewards=rewards, n=n, n_checkpoints=n_checkpoints)

fig = plt.figure(num=2)
ax = fig.add_subplot(111)
ax.plot(np.log10(vis[:, 0]), vis[:, 1])
plt.show()

# game_fig = plt.figure(num=4)
# game_ax = game_fig.add_subplot(111)
# run_game(states=states, ax=game_ax)
