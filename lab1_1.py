from lab1_functions import *
from lab1_classes import *


maze = generate_maze()
time_horizon = 16
entry = (0, 0)
exit = (6, 5)

rewards = Reward(reward_eaten=-1, reward_exiting=1)

states = StateSpace(mdp='finite', maze=maze,
                    time_horizon=time_horizon,
                    entry=entry, exit=exit,
                    minotaur_can_stand_still=False)

backward_induction(states=states, rewards=rewards)

show_values(states=states, maze=maze, minotaur_position=(5, 5), t=14)
show_policies(states=states, maze=maze, minotaur_position=(5, 5), t=14)

while True:
    run_finite_game(states=states)
    plt.show()

plt.show()


# states = StateSpace(mdp='discounted', maze=maze,
#                     lifetime_mean=30,
#                     precision=0.1,
#                     entry=entry, exit=exit,
#                     minotaur_can_stand_still=True)
# value_iteration(states=states, rewards=rewards, plot=False)
#
# run_game(states, rewards)
#
# show_values(states=states, maze=maze, minotaur_position=(4, 5))
# show_policies(states=states, maze=maze, minotaur_position=(4, 5))
#
# plt.show()

# '''


# lifetime_mean = 30
# lifetime_distribution = geom(p=1/lifetime_mean)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = np.r_[:150]
# ax.vlines(x, 0, lifetime_distribution.pmf(x), colors='k', linestyles='-', lw=1, label='frozen pmf')
# ax.legend(loc='best', frameon=False)
# plt.show()
# print(lifetime_distribution.pmf(2)/lifetime_distribution.pmf(1))
# print(29/30)