# from import-stuff import *
from lab0_functions import *

" PROBLEM 1 "

start_position = (0, 0)
target_position = (5, 5)
t_horizon = 40
discount = 0.95
precision = 1e-2

maze = generate_maze()

states = generate_maze_states(maze=maze, t_horizon=t_horizon, start=start_position, target=target_position)
rewards = generate_transition_rewards(states=states, target=State.target, reward_staying=0, reward_moving=-1, reward_target=1)
bi_values = backward_induction(maze=maze, states=states, t_horizon=t_horizon, rewards=rewards, plot=False, pause_time=0.1)
plot_most_rewarding_path(s0=State.start, values=bi_values, pause_time=0.1)


states = generate_maze_states2(maze=maze, start=start_position, target=target_position, discount=discount, precision=precision)
rewards = generate_transition_rewards(states=states, target=State2.target, reward_staying=0, reward_moving=-1, reward_target=1)
(vi_values, n) = value_iteration(maze=maze, states=states, rewards=rewards, plot=False, pause_time=0.1)
plot_most_rewarding_policy(states=states, values=vi_values, n=n, algorithm='Value Iteration')

states = generate_maze_states2(maze=maze, start=start_position, target=target_position, discount=discount, precision=precision)
rewards = generate_transition_rewards(states=states, target=State2.target, reward_staying=0, reward_moving=-1, reward_target=1)
(pi_values, n) = policy_iteration(maze=maze, states=states, rewards=rewards, plot=False, pause_time=0.1)
plot_most_rewarding_policy(states=states, values=pi_values, n=n, algorithm='Policy Iteration')

print('Backwards Induction solution')
print(bi_values)
print('Value Iteration solution')
print(vi_values)
print('Policy Iteration solution')
print(pi_values)

plt.show()
