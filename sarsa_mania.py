import numpy as np
from itertools import count
import copy
from numpy.random import choice
import matplotlib.pyplot as plt
import matplotlib.cm as matplotlib_colormap


class StateSpace:
    def __init__(self, grid_size, r, p, b, r_epsilon, p_epsilon):
        self.grid_size = grid_size
        self.robber_initial_position = r
        self.police_initial_position = p
        self.bank_position = b

        self.initial_state = None

        self.robber_epsilon = r_epsilon
        self.robber_states = []
        self.generate_robber_states()
        self.n_robber_states = len(self.robber_states)

        self.police_epsilon = p_epsilon
        self.police_states = []
        self.generate_police_states()
        self.n_police_states = len(self.police_states)

        self.states = []
        self.states_matrix = np.full((self.n_robber_states, self.n_police_states), State)
        self.generate_states()

    def __getitem__(self, i):
        return self.states[i]

    def __len__(self):
        return len(self.states)

    def generate_states(self):
        state_count = count(0)
        for robber in self.robber_states:
            for police in self.police_states:
                state = State(id=next(state_count),
                              robber=copy.deepcopy(robber),
                              police=copy.deepcopy(police))
                if robber.position == self.robber_initial_position and police.position == self.police_initial_position:
                    self.initial_state = state

                self.states.append(state)
                self.states_matrix[robber.id, police.id] = state

        for state in self.states:
            state.robber.initialize_q_values()
            state.police.initialize_q_values()

    def generate_robber_states(self):
        robber_count = count(0)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.robber_states.append(Robber(id=next(robber_count),
                                                 position=(i, j),
                                                 bank_position=self.bank_position,
                                                 epsilon=self.robber_epsilon))

        for state in self.robber_states:
            state.neighbours.append(state)  # can stand still
            for other_state in self.robber_states:
                if np.abs(state.position[0]-other_state.position[0]) == 1 and np.abs(state.position[1]-other_state.position[1]) == 0:
                    state.neighbours.append(other_state)    # column neighbour
                elif np.abs(state.position[0]-other_state.position[0]) == 0 and np.abs(state.position[1]-other_state.position[1]) == 1:
                    state.neighbours.append(other_state)    # row neighbour

    def generate_police_states(self):
        police_count = count(0)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.police_states.append(Police(id=next(police_count), position=(i, j),
                                                 epsilon=self.robber_epsilon))

        for state in self.police_states:
            state.neighbours.append(state)  # can stand still
            for other_state in self.police_states:
                if np.abs(state.position[0]-other_state.position[0]) == 1 and np.abs(state.position[1]-other_state.position[1]) == 0:
                    state.neighbours.append(other_state)    # column neighbour
                elif np.abs(state.position[0]-other_state.position[0]) == 0 and np.abs(state.position[1]-other_state.position[1]) == 1:
                    state.neighbours.append(other_state)    # row neighbour

    def state_where(self, robber=None, police=None):
        return self.states_matrix[robber.id, police.id]

    def subset_where(self, robber_id=None, police_id=None):
        if police_id is None and robber_id in range(self.n_robber_states):
            return list(self.states_matrix[robber_id, :])

        elif robber_id is None and police_id in range(self.n_police_states):
            return list(self.states_matrix[:, police_id])

        else:
            print('Error: Specify a valid id for a robber or a police state.')
            return None

    def possible_next_states(self, state, action):
        next_robber_state = state.robber.neighbours[action]
        subset = self.subset_where(robber_id=next_robber_state.id)
        return [subset[next_police.id] for next_police in state.police.neighbours]

    def reset_actions(self):
        for state in self.states:
            state.robber.action = 0


class RobberReward:
    def __init__(self):
        self.robbing_bank = 1
        self.being_caught = -1

    def __call__(self, state):
        if state.robber_caught():
            return self.being_caught
        elif state.robber_robbing():
            return self.robbing_bank
        else:
            return 0


class PoliceReward:
    def __init__(self):
        self.bank_robbed = 0
        self.catching_robber = 1

    def __call__(self, state):
        if state.robber_robbing():
            return self.bank_robbed
        elif state.robber_caught():
            return self.catching_robber
        else:
            return 0


class State:
    def __init__(self, id, robber, police):
        self.id = id
        self.robber = robber
        self.police = police

    def __str__(self):
        return str(self.robber) + '\n' + str(self.police)

    def robber_caught(self):
        if self.robber.position == self.police.position:
            return True
        else:
            return False

    def robber_robbing(self):
        if self.robber.is_in_bank() and not self.robber_caught():
            return True
        else:
            return False

#class Agent:
#    def __init__(self, id, position, bank_position):


class Robber:   # (Agent)
    def __init__(self, id, position, bank_position, epsilon):
        self.id = id
        self.position = position
        self.epsilon = epsilon
        self.neighbours = []
        self.action = 0
        self.bank_position = bank_position

        self.q_a = []

    def __str__(self):
        return 'robber @ ' + str(self.position)

    def initialize_q_values(self):
        for _ in self.neighbours:
            q = QValue()
            self.q_a.append(q)

    def is_in_bank(self):
        return self.position == self.bank_position

    def select_action(self):
        if choice(a=[True, False], size=1, p=[1-self.epsilon, self.epsilon]):     # greedy
            return self.action
        else:
            actions = list(np.r_[:len(self.neighbours)])
            actions.remove(self.action)
            return choice(actions)


class Police:   # (Agent)
    def __init__(self, id, position, epsilon):
        self.id = id
        self.epsilon = epsilon
        self.position = position
        self.neighbours = []
        self.action = 0
        self.q_a = []

    def __str__(self):
        return 'police @ ' + str(self.position)

    def initialize_q_values(self):
        for _ in self.neighbours:
            q = QValue()
            self.q_a.append(q)

    def select_action(self):
        if choice(a=[True, False], size=1, p=[1-self.epsilon, self.epsilon]):     # greedy
            return self.action
        else:
            actions = list(np.r_[:len(self.neighbours)])
            actions.remove(self.action)
            return choice(actions)


class QValue:   # action-state value
    def __init__(self):
        self.value = 0.0
        self.n_updates = 1

    def __call__(self):
        return self.value

    def __str__(self):
        return '{:.4f}'.format(self.value)

    def update_value(self, correction):
        self.value += self.n_updates ** (-2/3) * correction
        self.n_updates += 1


def run_game(states, ax):
    max_game_time = 50
    ax.clear()
    grid = np.ones(states.grid_size)
    grid[states.bank_position] = 0
    colormap = matplotlib_colormap.get_cmap('RdBu')
    ax.matshow(grid, cmap=colormap)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    state = states.initial_state
    (r2, r1) = state.robber.position
    (p2, p1) = state.police.position
    robber_mark, = ax.plot(r1, r2, color=colormap(0.15), marker='$R$', markersize=16)     # create mark for robber
    police_mark, = ax.plot(p1, p2, color=colormap(0.85), marker='$P$', markersize=16)     # create mark for police
    plt.pause(0.5)                          # pause and draw
    i = 0
    while i < max_game_time:
        i += 1
        state = states.state_where(robber=state.robber.neighbours[state.robber.select_action()],
                                   police=state.police.neighbours[state.police.select_action()])

        (r2, r1) = state.robber.position
        (p2, p1) = state.police.position
        robber_mark.set_data(r1, r2)
        police_mark.set_data(p1, p2)
        plt.pause(0.2)  # pause and update (draw) positions


def sarsa(agent, action, reward, next_agent, next_action, gamma=0.8):
    qt = agent.q_a[action]
    qt_next = next_agent.q_a[next_action]
    correction = reward + gamma * qt_next() - qt()
    qt.update_value(correction=correction)
    q_values = [qsa() for qsa in agent.q_a]
    agent.action = np.argmax(q_values)


grid_size = (4, 4)
robber_position = (0, 0)
police_position = (3, 3)
bank_position = (1, 1)
greedy_robber = 0.2
greedy_police = 0.2

states = StateSpace(grid_size=grid_size, r=robber_position, p=police_position, b=bank_position,
                    r_epsilon=greedy_robber, p_epsilon=greedy_police)

robber_rewards = RobberReward()
police_rewards = PoliceReward()

state = states.initial_state
n = 1000000

fig = plt.figure(num=10)
ax = fig.add_subplot(111)

for t in range(n):
#    if t % 100000 == 0 and t > n/3:
#        print(t)
#        print(t/n)
#        run_game(states=states, ax=ax)

    robber_reward = robber_rewards(state=state)
    police_reward = police_rewards(state=state)

    robber = state.robber
    police = state.police
    robber_action = robber.select_action()
    police_action = police.select_action()

    next_state = states.state_where(robber=robber.neighbours[robber_action], police=police.neighbours[police_action])
    next_robber = next_state.robber
    next_police = next_state.police

    sarsa(agent=robber, action=robber_action, reward=robber_reward, next_agent=next_robber, next_action=next_robber.action)
    sarsa(agent=police, action=police_action, reward=police_reward, next_agent=next_police, next_action=next_police.action)

    state = next_state


while True:
    run_game(states=states, ax=ax)
