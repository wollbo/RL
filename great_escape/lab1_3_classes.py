import numpy as np
from itertools import count
import copy
from numpy.random import choice


class StateSpace:
    def __init__(self, grid_size, r, p, b):
        self.grid_size = grid_size
        self.robber_initial_position = r
        self.police_initial_position = p
        self.bank_position = b

        self.initial_state = None
        self.discount_factor = 0.8

        self.robber_states = []
        self.generate_robber_states()
        self.n_robber_states = len(self.robber_states)

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
                              police=police)
                if robber.position == self.robber_initial_position and police.position == self.police_initial_position:
                    self.initial_state = state

                self.states.append(state)
                self.states_matrix[robber.id, police.id] = state

    def generate_robber_states(self):
        robber_count = count(0)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.robber_states.append(Robber(id=next(robber_count),
                                                 position=(i, j),
                                                 bank_position=self.bank_position))

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
                self.police_states.append(Police(id=next(police_count), position=(i, j)))

        for state in self.police_states:
            # state.neighbours.append(state) cannot stand still
            for other_state in self.police_states:
                if np.abs(state.position[0]-other_state.position[0]) == 1 and np.abs(state.position[1]-other_state.position[1]) == 0:
                    state.neighbours.append(other_state)    # column neighbour
                elif np.abs(state.position[0]-other_state.position[0]) == 0 and np.abs(state.position[1]-other_state.position[1]) == 1:
                    state.neighbours.append(other_state)    # row neighbour

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

class Reward:
    def __init__(self):
        self.robbing_bank = 1
        self.being_caught = -10

    def __call__(self, state):
        if state.robber_caught():
            return self.being_caught
        elif state.robber_robbing():
            return self.robbing_bank
        else:
            return 0


class State:
    def __init__(self, id, robber, police):
        self.id = id
        self.robber = robber
        self.police = police
        #self.value = 0          # v0

    def __str__(self):
        return str(self.robber) + '\n' + str(self.police)

    def robber_caught(self):
        if self.robber.position == self.police.position:
            return True
        else:
            return False

    def robber_robbing(self):
        if not self.robber_caught() and self.robber.is_in_bank():
            return True
        else:
            return False


class Robber:
    def __init__(self, id, position, bank_position):
        self.id = id
        self.position = position
        self.neighbours = []
        self.action = 0
        self.bank_position = bank_position

    def __str__(self):
        return 'robber @ ' + str(self.position)

    def is_in_bank(self):
        return self.position == self.bank_position

    def select_action(self, epsilon):
        if choice(a=[True, False], size=1, p=[1-epsilon, epsilon]):     # greedy
            return self.action
        else:
            actions = list(np.r_[:len(self.neighbours)])
            actions.remove(self.action)
            return choice(actions)


class Police:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.neighbours = []

    def __str__(self):
        return 'police @ ' + str(self.position)


class QFunction:
    def __init__(self, states):
        self.states = states
        self.discount_factor = states.discount_factor
        self.q_values = []
        for state in states:
            qs = []
            for action in range(len(state.robber.neighbours)):
                qsa = QValue()
                qs.append(qsa)
            self.q_values.append(qs)

    def __str__(self):
        s = 'actions: \t0\t1\t2\t3\t4\n'
        for i in range(len(self.q_values)):
            qsas = self.q_values[i]
            nqsas = len(qsas)
            s += 'state {:.0f} \t'.format(i)
            for j in range(nqsas):
                s += str(qsas[j]) + '\t'
            s += '\n'
        return s

    def __call__(self, state, action):
        return self.__internal_call__(state=state, action=action)

    def __internal_call__(self, state, action):
        return self.q_values[state.id][action]

    def update(self, state, action, reward, next_state):
        qt = self.__internal_call__(state=state, action=action)
        next_state_q_values = [qsb.value for qsb in self.q_values[next_state.id]]
        correction = reward + self.discount_factor * np.max(next_state_q_values) - qt.value
        qt.update_value(correction=correction)

    def update_sarsa(self, state, action, reward, next_state, next_action):
        qt = self.__internal_call__(state=state, action=action)
        next_state_action = self.__internal_call__(state=next_state, action=next_action)
        correction = reward + self.discount_factor * next_state_action.value - qt.value
        qt.update_value(correction=correction)
        q_values_in_state = [qsa.value for qsa in self.q_values[state.id]]
        state.robber.action = np.argmax(q_values_in_state)

    def set_policies(self):
        for state in self.states:
            q_values = [qsa.value for qsa in self.q_values[state.id]]
            state.robber.action = np.argmax(q_values)

    def values_when(self, police_position=(0, 0)):
        mat = np.zeros(self.states.grid_size)
        police = [p for p in self.states.police_states if p.position == police_position][0]

        state_subset = self.states.subset_where(police_id=police.id)
        for state in state_subset:
            mat[state.robber.position] = self.q_values[state.id][state.robber.action].value
        return mat


class QValue:   # action-state value
    def __init__(self):
        self.value = 0.0
        self.n_updates = 1

    def __str__(self):
        return '{:.4f}'.format(self.value)

    def update_value(self, correction):
        self.value += self.n_updates ** (-2/3) * correction
        self.n_updates += 1



