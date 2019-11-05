# from import-stuff import *
from lab0_functions import *

n_rows = 6
n_cols = 7
maze = np.full((n_rows, n_cols), True)
maze[0:3, 2] = False
maze[4, 1:6] = False

# maze[7, 3:7] = False
# maze[7:14, 3] = False
# maze[7:14, 7] = False

t_horizon = 30
pause_time = 1

states = generate_maze_states2(maze=maze, th=t_horizon)
rewards = get_transition_reward2(states)

" backwards induction "
fig = plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for t in range(t_horizon-1, -1, -1):
    # print(t)
    # print(s.action[t])
    for s in states:
        possible_actions = [states.index(ne) for ne in s.neighbours]
        possible_rewards = rewards[states.index(s), possible_actions]

        if t == t_horizon-1:
            pass
        else:
            possible_rewards = possible_rewards + [ne.value[t+1] for ne in s.neighbours]

        s.action[t] = np.argmax(possible_rewards)
        s.value[t] = possible_rewards[s.action[t]]

    print(s)
    print(s.value[t])
    print(s.action[t])

    values = get_state_values2(states, t, n_rows, n_cols)

    ax1.matshow(np.log(values))
    ax1.set_xlabel('State values')
    show_state_policies2(states, t, ax2)
    ax2.set_xlabel('State policies at time {:.0f}'.format(t))
    plt.gca().set_aspect('equal')
    plt.axis([-1, n_cols, -n_rows, 1])
    plt.pause(pause_time)


plt.show()


# pt = get_transition_probability(states=states)
# pr = get_transition_reward(states=states)

# fig1 = plt.figure(1)
# fig1.suptitle('Policy value iteration')
# State.value_tolerance = 1e-1
# policy_value_iteration_infinite(maze=maze, states=states, transition_probability=pt, transition_reward=pr, gamma=0.9, pause_time=0.08)

# fig2 = plt.figure(2)
# fig2.suptitle('Value iteration')
# states = generate_maze_states(maze=maze)
# State.value_tolerance = 1e-4
# value_iteration_infinite(maze=maze, states=states, transition_probability=pt, transition_reward=pr, gamma=0.9, pause_time=0.08)

# fig3 = plt.figure(3)
# fig3.suptitle('Modified Problem')
# states = generate_maze_states(maze=maze)
# State.value_tolerance = 1e-5
# value_iteration(maze=maze, states=states, gamma=0.9, pause_time=0.08)

#plt.show()
