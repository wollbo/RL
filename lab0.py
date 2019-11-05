# from import-stuff import *
from lab0_functions import *

start_position = (0, 0)
target_position = (5, 5)
t_horizon = 10

maze = generate_maze()

states = generate_maze_states(maze=maze, th=t_horizon, start_position=start_position, target_position=target_position)

rewards = get_transition_reward(states, reward_staying=-1, reward_moving=-1, reward_target=100)

backwards_induction(maze, states, t_horizon, rewards, pause_time=0.1)

plot_shortest_path(start=State.start_state, t_horizon=t_horizon, maze=maze, pause_time=0.2)

plt.show()

# pt = get_transition_probability0(states=states)
# pr = get_transition_reward0(states=states)

# fig1 = plt.figure(1)
# fig1.suptitle('Policy value iteration')
# State0.value_tolerance = 1e-1
# policy_value_iteration_infinite(maze=maze, states=states, transition_probability=pt, transition_reward=pr, gamma=0.9, pause_time=0.08)

# fig2 = plt.figure(2)
# fig2.suptitle('Value iteration')
# states = generate_maze_states0(maze=maze)
# State0.value_tolerance = 1e-4
# value_iteration_infinite(maze=maze, states=states, transition_probability=pt, transition_reward=pr, gamma=0.9, pause_time=0.08)

# fig3 = plt.figure(3)
# fig3.suptitle('Modified Problem')
# states = generate_maze_states(maze=maze)
# State0.value_tolerance = 1e-5
# value_iteration(maze=maze, states=states, gamma=0.9, pause_time=0.08)

#plt.show()


# '''