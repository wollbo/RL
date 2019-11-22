from lab1_3_classes import *
from lab1_3_functions import *
from numpy.random import choice

grid_size = (4, 4)
robber_position = (0, 0)
police_position = (3, 3)
bank_position = (1, 1)

states = StateSpace(grid_size=grid_size,
                    r=robber_position,
                    p=police_position,
                    b=bank_position)
rewards = Reward()

q_function = QFunction(states=states)

state = states.initial_state
n = 1000
for t in range(n):
    possible_actions = np.r_[:len(state.robber.neighbours)]
    action = choice(possible_actions)
    reward = rewards(state=state)
    next_state = choice(states.possible_next_states(state=state, action=action))
    q_function.update(state=state, action=action, reward=reward, next_state=next_state)
    state = next_state

    if t % 1000 == 0:
        print(t/n)

q_function.set_policies()

state = states.initial_state

fig = plt.figure()
ax = fig.add_subplot(111)
run_game(states=states, ax=ax)

# for _ in range(100):
#     if state.robber_caught():
#         print('robber caught')
#     elif state.robber_robbing():
#         print('money')
#     else:
#         print('moving around')
#
#     reward = rewards(state=state)
#     action = state.robber.action
#     possible_next_states = states.possible_next_states(state=state, action=action)
#     next_state = choice(possible_next_states)
#     state = next_state

# '''
