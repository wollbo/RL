from lab0_functions import *

" PROBLEM 2 "

maze = np.full((6, 7), True)
(n_rows, n_cols) = maze.shape

weights = array([[0, 1, -np.inf, 10, 10, 10, 10],
                 [0, 1, -np.inf, 10, 0, 0, 10],
                 [0, 1, -np.inf, 10, 0, 0, 10],
                 [0, 1, 1, 1, 0, 0, 0],
                 [0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 10],
                 [0, 0, 0, 0, 0, 11, 10]])


for t_horizon in range(38, 39):
    pass
    # states = generate_maze_states(maze=maze, t_horizon=t_horizon, start=(0, 0))
    # rewards = generate_transitions_rewards2(weights=weights, states=states)
    # bi_values = backward_induction(maze=maze, states=states, rewards=rewards, t_horizon=t_horizon, plot=False, pause_time=0.1)
    # plot_most_rewarding_path(State.start, values=bi_values, pause_time=0.01)


states = generate_maze_states2(maze=maze, start=(0, 0))
rewards = generate_transitions_rewards2(weights=weights, states=states)
(vi_values, gamma, n_iterations) = value_iteration(maze=maze, states=states, rewards=rewards, gamma=0.99, plot=False, pause_time=0.1)
plot_most_rewarding_policy(states=states, values=vi_values, gamma=gamma, n=n_iterations)

print(bi_values)
print(vi_values)

plt.show()
