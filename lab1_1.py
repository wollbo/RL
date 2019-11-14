from lab1_functions import *
from lab1_classes import *


maze = generate_maze()
time_horizon = 20
entry = (0, 0)
exit = (6, 5)

states = StateSpace(maze=maze,
                    time_horizon=time_horizon,
                    entry=entry, exit=exit,
                    minotaur_can_stand_still=True)

rewards = Reward(reward_eaten=-100, reward_exiting=100)

backward_induction(states=states, rewards=rewards)

run_game(maze=maze, states=states)

show_policies(states=states, maze=maze, t=3, minotaur_position=(4, 4))

plt.show()
