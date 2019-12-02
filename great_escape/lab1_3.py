from great_escape.lab1_3_functions import *

grid_size = (4, 4)
robber_position = (0, 0)
police_position = (3, 3)
bank_position = (1, 1)

states = StateSpace(grid_size=grid_size,
                    r=robber_position,
                    p=police_position,
                    b=bank_position)
rewards = Reward()

n = 1000000          # 00
n_checkpoints = 1000
game_times = np.int_(n*np.array([0.5, 0.7, 0.9]))
# vis = q_learning_algorithm(states=states, rewards=rewards, n=n, n_checkpoints=n_checkpoints, plot=True, game_times=game_times)
q_learning_algorithm(states=states, rewards=rewards, n=n, n_checkpoints=n_checkpoints)


# np.savetxt(X=vis, fmt=['%10.0f', '%10.2e'], fname='eis_q_iterative.txt')
# fig = plt.figure(num=2)
# ax = fig.add_subplot(111)
# ax.plot(vis[:, 0], vis[:, 1])
# plt.show()

# game_fig = plt.figure(num=4)
# game_ax = game_fig.add_subplot(111)
# run_game(states=states, ax=game_ax)

game_fig = plt.figure(num=4)
game_ax = game_fig.add_subplot(111)
run_game(states=states, ax=game_ax)

show_policies(states=states, police_position=(3, 3), fignum=10)
show_policies(states=states, police_position=(2, 3), fignum=11)
show_policies(states=states, police_position=(3, 2), fignum=12)
show_policies(states=states, police_position=(2, 2), fignum=13)
plt.show()


# '''