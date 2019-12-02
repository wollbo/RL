from great_escape.lab0_functions import *
from great_escape.lab0_classes import *

" PROBLEM 2 "

maze = np.full((6, 7), True)
(n_rows, n_cols) = maze.shape

start_position = (0, 0)
target_position = (5, 5)
r1_position = (5, 0)
r2_position = (3, 6)
time_horizon = 40
discount = 0.99
precision = 0.3

weights = array([[0, 1, -np.inf, 10, 10, 10, 10],
                 [0, 1, -np.inf, 10, 0, 0, 10],
                 [0, 1, -np.inf, 10, 0, 0, 10],
                 [0, 1, 1, 1, 0, 0, 0],
                 [0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 10],
                 [0, 0, 0, 0, 0, 11, 10]])


# for t_horizon in range(38, 39):
states = StateSpace(maze=maze, time_horizon=time_horizon, start=start_position)
rewards = Reward2(weights=weights, states=states)
bi_values = backward_induction(maze=maze, states=states, rewards=rewards, plot=False, pause_time=0.5)
plot_most_rewarding_path(states=states, pause_time=0.01)

states = StateSpace2(maze=maze, precision=precision, discount=discount, start=start_position)
rewards = Reward2(weights=weights, states=states)
(vi_values, n_iterations) = value_iteration(maze=maze, states=states, rewards=rewards, plot=False, pause_time=0.1)
plot_most_rewarding_policy(states=states, n=n_iterations, algorithm='Value Iteration')

states = StateSpace2(maze=maze, precision=precision, discount=discount, start=start_position)
rewards = Reward2(weights=weights, states=states)
(pi_values, n_iterations) = policy_iteration(maze=maze, states=states, rewards=rewards, plot=False, pause_time=0.1)
plot_most_rewarding_policy(states=states, n=n_iterations, algorithm='Policy Iteration')


# #print('Backwards Induction solution')
# #print(bi_values)
# print('Value Iteration solution')
# print(vi_values)
# print('Policy Iteration solution')
# print(pi_values)

plt.show()
