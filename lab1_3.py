from lab1_3_classes import *
from numpy.random import choice

grid_size = (4, 4)
states = StateSpace(grid_size=grid_size)
rewards = Reward()

q_function = QFunction(states=states)

state = states.initial_state


for t in range(100):
    print(t)
    reward = rewards(state=state)
    print(reward)
    possible_actions = np.r_[:len(state.robber.neighbours)]
    action = choice(possible_actions)
    possible_next_states = states.possible_next_states(state=state, action=action)
    next_state = choice(possible_next_states)

    q_function.update(state=state, action=action, reward=reward, next_state=next_state)

# print(q_function)

for qs in q_function.q_values:
    for q in qs:
        print(q)

