# from import-stuff import *
from lab0_functions import *

n_rows = 6
n_cols = 7
maze = np.full((n_rows, n_cols), True)
maze[0:3, 2] = False
maze[4, 1:6] = False

# maze[7, 3:7] = False
# maze[7:14, 3] = False
# maze[7:14, 7] = False

states = generate_states(maze=maze)
fig1 = plt.figure(1)
fig1.suptitle('Policy value iteration')
State.value_tolerance = 1e-2
policy_value_iteration(maze=maze, states=states, gamma=0.9, pause_time=0.08)

fig2 = plt.figure(2)
fig2.suptitle('Value iteration')
State.value_tolerance = 1e-5
value_iteration(maze=maze, states=states, gamma=0.9, pause_time=0.08)

plt.show()
