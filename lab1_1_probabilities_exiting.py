from lab1_functions import *
from lab1_classes import *

maze = generate_maze()
entry = (0, 0)
exit = (6, 5)
time_horizon = 20

n = 10000

file = open('exiting_probailities.txt', 'w+')

rewards = Reward(reward_eaten=-1, reward_exiting=1)

T0 = 15
p1 = []
# horizons = range(T0, 20)
# for time_horizon in horizons:
#     states = StateSpace(mdp='finite',
#                         maze=maze,
#                         time_horizon=time_horizon,
#                         entry=entry, exit=exit,
#                         minotaur_can_stand_still=False)
#     backward_induction(states=states, rewards=rewards)
#     file.write(states.get_info())
#     p = probability_of_exiting(states=states, n_tests=n)
#     file.write('\nprobability of exiting (average from {:.0f} trials) is {:.8f}\n'.format(n, p))
#     p1.append(p)

p1 = []
horizons = range(30, 41)
for time_horizon in horizons:
    states = StateSpace(mdp='finite',
                        maze=maze,
                        time_horizon=time_horizon,
                        entry=entry, exit=exit,
                        minotaur_can_stand_still=True)
    backward_induction(states=states, rewards=rewards)
    file.write(states.get_info())
    p = probability_of_exiting(states=states, n_tests=n)
    file.write('\nprobability of exiting (average from {:.0f} trials) is {:.8f}'.format(n, p))
    p1.append(p)
file.write('\n' + str(p1))

# states = StateSpace(mdp='discounted',
#                     maze=maze,
#                     lifetime_mean=30,
#                     precision=0.01,
#                     entry=entry, exit=exit,
#                     minotaur_can_stand_still=False)
# value_iteration(states=states, rewards=rewards, plot=False)
# file.write(states.get_info())
# p = probability_of_exiting(states=states, n_tests=n)
# file.write('\nprobability of exiting (average from {:.0f} trials) is {:.8f}'.format(n, p))
#
#
# states = StateSpace(mdp='discounted',
#                     maze=maze,
#                     lifetime_mean=30,
#                     precision=0.01,
#                     entry=entry, exit=exit,
#                     minotaur_can_stand_still=True)
# value_iteration(states=states, rewards=rewards, plot=False)
# file.write(states.get_info())
# p = probability_of_exiting(states=states, n_tests=n)
# file.write('\nprobability of exiting (average from {:.0f} trials) is {:.8f}'.format(n, p))


#while True:
#    run_finite_game(states=states)
#plt.show()


# file.write(states.get_info())
# p = probability_of_exiting(states=states, n_tests=n)
# file.write('\nprobability of exiting (average from {:.0f} trials) is {:.8f}'.format(n, p))
#
#
# states = StateSpace(mdp='discounted',
#                     maze=maze, lifetime_mean = 30,
#                     precision=0.1,
#                     entry=entry, exit=exit,
#                     minotaur_can_stand_still=True)
#
# value_iteration(states=states, rewards=rewards, plot=False)
#
# p = probability_of_exiting(states=states, n_tests=n)


file.close()