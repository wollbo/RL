# from import-stuff import *
from lab0_functions import *
from lab0_classes import *

" PROBLEM 1 "

start_position = (0, 0)
target_position = (5, 5)
r1_position = (5, 0)
r2_position = (3, 6)
time_horizon = 12
discount = 0.9
precision = 1
maze = generate_maze()


for _ in range(1):
    states = StateSpace(maze=maze, time_horizon=time_horizon, start=start_position, target=target_position, r1=r1_position, r2=r2_position)
    rewards = Reward(states, reward_target=3, reward_staying=-1, reward_moving=-1)
    bi_values = backward_induction(maze=maze, states=states, rewards=rewards, plot=True, pause_time=0.1)
    plot_most_rewarding_path(states=states, pause_time=0.1)

states = StateSpace2(maze=maze, discount=discount, precision=precision, start=start_position, target=target_position, r1=r1_position, r2=r2_position)
rewards = Reward(states, reward_target=12, reward_staying=0, reward_moving=-1)
(vi_values, n) = value_iteration(maze=maze, states=states, rewards=rewards, plot=True, pause_time=0.1)
plot_most_rewarding_policy(states=states, n=n, algorithm='Value Iteration')

states = StateSpace2(maze=maze, discount=discount, precision=precision, start=start_position, target=target_position, r1=r1_position, r2=r2_position)
rewards = Reward(states, reward_target=12, reward_staying=0, reward_moving=-1)
(pi_values, n) = policy_iteration(maze=maze, states=states, rewards=rewards, plot=True, pause_time=0.1)
plot_most_rewarding_policy(states=states, n=n, algorithm='Policy Iteration')


# print('Backwards Induction solution')
# print(bi_values)
# print('Value Iteration solution')
# print(vi_values)
# print('Policy Iteration solution')
# print(pi_values)

plt.show()
