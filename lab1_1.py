from lab1_functions import *
from lab1_classes import *


maze = generate_maze()
time_horizon = 20
entry = (0, 0)
exit = (6, 5)
rewards = Reward(reward_eaten=-1, reward_exiting=1)

# states = StateSpace(mdp='finite',
#                     maze=maze,
#                     time_horizon=time_horizon,
#                     entry=entry, exit=exit,
#                     minotaur_can_stand_still=False)
#
#
# backward_induction(states=states, rewards=rewards)
# v0 = states.values(t=0)

states = StateSpace(mdp='discounted',
                    maze=maze,
                    discount_factor=0.95,
                    precision=1,
                    entry=entry, exit=exit,
                    minotaur_can_stand_still=False)

# states.set_initial_values(v0=v0)

n = value_iteration(states=states, rewards=rewards, plot=False)
show_values(states=states, maze=maze, minotaur_position=(4, 5))
show_policies(states=states, maze=maze, minotaur_position=(4, 5))

#while True:
    # run_game(maze=maze, states=states, rewards=rewards)

#show_policies(states=states, maze=maze, t=15, minotaur_position=(6, 5))
# show_policies(states=states, maze=maze, t=16, minotaur_position=(6, 5))
# show_policies(states=states, maze=maze, t=17, minotaur_position=(6, 5))
# show_policies(states=states, maze=maze, t=18, minotaur_position=(6, 5))
# show_policies(states=states, maze=maze, t=19, minotaur_position=(6, 5))

#show_values(states=states, maze=maze, t=17, minotaur_position=(6, 5))

plt.show()

# while True:
#    run_game(maze=maze, states=states)
# plt.show()

# '''