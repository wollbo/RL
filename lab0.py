# from import-stuff import *
from lab0_functions import *
from lab0_classes import *

" PROBLEM 1 "

start_position = (0, 0)
target_position = (5, 5)
t_horizon = 12
discount = 0.95
precision = 1
maze = generate_maze()

for _ in range(10):
    states = generate_maze_states(maze=maze, t_horizon=t_horizon, start=start_position, target=target_position, r1=(5, 0), r2=(3, 6))
    rewards = Reward(states, target=State.target, reward_target=3, reward_staying=-1, reward_moving=-1, r1=State.r1, r2=State.r2)
    bi_values = backward_induction(maze=maze, states=states, t_horizon=t_horizon, rewards=rewards, plot=False, pause_time=0.1)
    plot_most_rewarding_path(s0=State.start, values=bi_values, pause_time=0.1)

# states = generate_maze_states2(maze=maze, start=start_position, target=target_position, discount=discount, precision=precision, r1=(5, 0), r2=(3, 6))
# rewards = Reward(states, target=State2.target, reward_target=12, reward_staying=0, reward_moving=-1, r1=State2.r1, r2=State2.r2)
# (vi_values, n) = value_iteration(maze=maze, states=states, rewards=rewards, plot=False, pause_time=0.1)
# plot_most_rewarding_policy(states=states, values=vi_values, n=n, algorithm='Value Iteration')

for _ in range(10):
    states = generate_maze_states2(maze=maze, start=start_position, target=target_position, discount=discount, precision=precision, r1=(5, 0), r2=(3, 6))
    rewards = Reward(states, target=State2.target, reward_target=12, reward_staying=0, reward_moving=-1, r1=State2.r1, r2=State2.r2)
    (pi_values, n) = policy_iteration(maze=maze, states=states, rewards=rewards, plot=False, pause_time=0.1)
    plot_most_rewarding_policy(states=states, values=pi_values, n=n, algorithm='Policy Iteration')

# print('Backwards Induction solution')
# print(bi_values)
# print('Value Iteration solution')
# print(vi_values)
# print('Policy Iteration solution')
# print(pi_values)

plt.show()
