# from import-stuff import *
from lab0_functions import *

" PROBLEM 1 "

start_position = (0, 0)
target_position = (5, 5)
t_horizon = 12

maze = generate_maze()

states = generate_maze_states(maze=maze, t_horizon=t_horizon, start=start_position, target=target_position)

rewards = generate_transition_rewards(states=states, target=State.target, reward_staying=0, reward_moving=-1, reward_target=12)

bi_values = backward_induction(maze=maze, states=states, t_horizon=t_horizon, rewards=rewards, pause_time=0.1)

# plot_most_rewarding_path(start=State.start, maze=maze, pause_time=0.1)

states = generate_maze_states2(maze=maze, start=start_position, target=target_position)

rewards = generate_transition_rewards(states=states, target=State2.target, reward_staying=0, reward_moving=-1, reward_target=12)

vi_values = value_iteration(maze=maze, states=states, rewards=rewards, gamma=0.8, pause_time=0.1)

print('Backwards Induction solution')
print(bi_values)
print('Value Iteration solution')
print(vi_values)

# states = generate_maze_states2(maze=maze, start=start_position, target=target_position)
# rewards = generate_transition_rewards(states=states, target=State2.target_state, reward_staying=0, reward_moving=-1, reward_target=12)
# pvi_values = policy_value_iteration(maze=maze, states=states, rewards=rewards, gamma=0.8, pause_time=0.1)
# print('Policy Value Iteration solution')
# print(pvi_values)

plt.show()
