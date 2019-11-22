from lab1_3_classes import *
from lab1_3_functions import *
from numpy.random import choice
import matplotlib.pyplot as plt


grid_size = (4, 4)
robber_position = (0, 0)
police_position = (3, 3)
bank_position = (1, 1)

grid_size = (4, 4)
robber_position = (0, 0)
police_position = (2, 2)
bank_position = (1, 1)

states = StateSpace(grid_size=grid_size,
                    r=robber_position,
                    p=police_position,
                    b=bank_position)
rewards = Reward()

q_function = QFunction(states=states)

state = states.initial_state

epsilon = 0.1

n = 10000000
accumulated_reward = np.zeros(n)
reward_t = np.empty(n)
fig = plt.figure()
ax = fig.add_subplot(111)

game_fig = plt.figure()
game_ax = game_fig.add_subplot(111)

for t in range(n):

    # if state.robber_caught():
    #     print('robber caught')
    # elif state.robber_robbing():
    #     print('\t\t\t\t\t\t\tmoney')
    #else:
    #    print('\t\t\tmoving around')

    reward = rewards(state=state)
    reward_t[t] = reward
    accumulated_reward[t] = accumulated_reward[t-1] + reward

    action = state.robber.select_action(epsilon=epsilon)
    next_robber_state = state.robber.neighbours[action]
    possible_next_states = states.subset_where(robber_id=next_robber_state.id)
    next_state = choice(possible_next_states)
    next_action = next_state.robber.action
    q_function.update_sarsa(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action)

    if t % 1000 == 0:
        print(t/n)
        #qt = q_function(state=state, action=action)
        #next_state_q_values = [qsb.value for qsb in q_function.q_values[next_state.id]]
        #print(next_state_q_values)
        pos = ax.matshow(q_function.values_when(police_position=(1, 1)))
        cb = fig.colorbar(pos, ax=ax, fraction=0.04)
        plt.pause(0.3)
        plt.cla()
        cb.remove()

        if t > 0.9*n:
            q_function.set_policies()
            run_game(states=states, ax=game_ax)

    state = next_state


print(accumulated_reward)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(accumulated_reward[:-1:10000])
plt.show()
