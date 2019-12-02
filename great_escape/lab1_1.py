from great_escape.lab1_functions import *
from great_escape.lab1_classes import *


maze = generate_maze()
entry = (0, 0)
exit = (6, 5)

rewards = Reward(reward_eaten=-1, reward_exiting=1)

# time_horizon = 17
#
# states = StateSpace(mdp='finite', maze=maze,
#                     time_horizon=time_horizon,
#                     entry=entry, exit=exit,
#                     minotaur_can_stand_still=True)
# backward_induction(states=states, rewards=rewards)
#
# show_values(states=states, maze=maze, minotaur_position=(5, 6), t=12)
# show_policies(states=states, maze=maze, minotaur_position=(5, 6), t=12)
#
# plt.show()

'''

while True:
    run_finite_game(states=states)
    plt.show()

plt.show()

# '''

states = StateSpace(mdp='discounted', maze=maze,
                    lifetime_mean=30,
                    precision=0.1,
                    entry=entry, exit=exit,
                    minotaur_can_stand_still=True)
value_iteration(states=states, rewards=rewards, plot=False)

run_finite_game(states=states)
run_finite_game(states=states)
run_finite_game(states=states)
run_finite_game(states=states)
run_finite_game(states=states)
run_finite_game(states=states)


show_values(states=states, maze=maze, minotaur_position=(5, 6))
show_policies(states=states, maze=maze, minotaur_position=(5, 6))

plt.show()
