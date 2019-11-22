from lab1_functions import *
from lab1_classes import *


maze = generate_maze()
time_horizon = 16
entry = (0, 0)
exit = (6, 5)

rewards = Reward(reward_eaten=-1, reward_exiting=1)

states = StateSpace(mdp='finite',
                    maze=maze,
                    time_horizon=time_horizon,
                    entry=entry, exit=exit,
                    minotaur_can_stand_still=True)
backward_induction(states=states, rewards=rewards)

run_game(states=states, rewards=rewards)


states = StateSpace(mdp='discounted',
                    maze=maze,
                    discount_factor=29/30,
                    precision=0.1,
                    entry=entry, exit=exit,
                    minotaur_can_stand_still=True)
value_iteration(states=states, rewards=rewards, plot=False)

run_game(states, rewards)

show_values(states=states, maze=maze, minotaur_position=(4, 5))
show_policies(states=states, maze=maze, minotaur_position=(4, 5))

plt.show()

# '''