# from import-stuff import *
from lab0_functions import *

n_rows = 6
n_cols = 7
maze = np.full((n_rows, n_cols), True)
maze[0:3, 2] = False
maze[4, 1:6] = False

# maze[7, 3:7] = False
# maze[7:9, 3] = False
# maze[7:9, 7] = False

states = generate_states(maze=maze)
# iterative_value_policy_update(maze=maze, states=states, gamma=0.9, pause_time=0.001)

value_iteration(maze=maze, states=states, gamma=0.9, pause_time=0.001)

