from numpy.random import choice
import numpy as np
from itertools import count
from great_escape.lab0_functions import plot_arrow


class StateSpace:       # state space for finite horizon problems
    def __init__(self, maze, time_horizon,
                 start=(None, None), target=(None, None),
                 r1=(None, None), r2=(None, None)):
        (n_rows, n_cols) = maze.shape
        self.maze_shape = (n_rows, n_cols)
        self.time_horizon = time_horizon
        self.states = []
        self.state_count = count(0)     # itertools count

        for row in range(n_rows):
            for col in range(n_cols):
                if maze[row, col]:
                    state = State(id=next(self.state_count),
                                  index=(row, col),
                                  time_horizon=self.time_horizon)
                    if (row, col) == start:
                        self.start = state
                    elif (row, col) == target:
                        self.target = state
                    elif (row, col) == r1:
                        self.r1 = state
                    elif (row, col) == r2:
                        self.r2 = state
                    self.states.append(state)

        for state in self.states:
            state.neighbours.append(state)  # s can go to s
            for other_state in self.states:
                if np.abs(other_state.index[0] - state.index[0]) == 1 and np.abs(
                        other_state.index[1] - state.index[1]) == 0:
                    state.neighbours.append(other_state)  # same row neighbours
                elif np.abs(other_state.index[0] - state.index[0]) == 0 and np.abs(
                        other_state.index[1] - state.index[1]) == 1:
                    state.neighbours.append(other_state)  # same column neighbours

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index]

    def values(self, t):
        vm = np.full(self.maze_shape, np.nan)
        for state in self.states:
            vm[state.index] = state.value[t]
        return vm

    def show_policies(self, t, axis):
        axis.clear()
        axis.set_yticklabels([])
        axis.set_xticklabels([])
        for state in self.states:
            other_state = state.neighbours[state.action[t]]
            plot_arrow(state, other_state)


class State:  # state for finite horizon problems
    def __init__(self, id, index, time_horizon):
        self.id = id
        self.index = index
        self.neighbours = []
        self.value = np.zeros(time_horizon)
        self.action = np.zeros(time_horizon, dtype=np.uint8)  # local index of best neighbour to go to at time t

    def __str__(self):
        return str(self.index)


class StateSpace2:      # state space for infinite horizon problems
    def __init__(self, maze, precision, discount,
                 start=(None, None), target=(None, None),
                 r1=(None, None), r2=(None, None)):
        (n_rows, n_cols) = maze.shape
        self.maze_shape = (n_rows, n_cols)
        self.states = []
        self.precision = precision  # epsilon
        self.discount = discount    # lambda
        self.diff_tolerance = self.precision * (1 - self.discount) / self.discount  # delta
        self.state_count = count(0)     # itertools count

        for row in range(n_rows):
            for col in range(n_cols):
                if maze[row, col]:
                    state = State2(id=next(self.state_count),
                                   index=(row, col))
                    if (row, col) == start:
                        self.start = state
                    elif (row, col) == target:
                        self.target = state
                    elif (row, col) == r1:
                        self.r1 = state
                    elif (row, col) == r2:
                        self.r2 = state
                    self.states.append(state)

        for state in self.states:
            state.neighbours.append(state)  # s can go to s
            for other_state in self.states:
                if np.abs(other_state.index[0] - state.index[0]) == 1 and np.abs(
                        other_state.index[1] - state.index[1]) == 0:
                    state.neighbours.append(other_state)  # same row neighbours
                elif np.abs(other_state.index[0] - state.index[0]) == 0 and np.abs(
                        other_state.index[1] - state.index[1]) == 1:
                    state.neighbours.append(other_state)  # same column neighbours

        self.iteration_difference = np.full(len(self), 1, dtype=np.float)

    def __getitem__(self, index):
        return self.states[index]

    def __len__(self):
        return len(self.states)

    def update_values(self):
        for state in self.states:
            self.iteration_difference[state.id] = state.update_value()

    def stopping_condition(self):
        return all(np.less(self.iteration_difference, self.diff_tolerance))

    def values(self):
        vm = np.full(self.maze_shape, np.nan)
        for state in self.states:
            vm[state.index] = state.value
        return vm

    def show_policies(self, axis):
        axis.clear()
        axis.set_yticklabels([])
        axis.set_xticklabels([])
        for state in self.states:
            other_state = state.neighbours[state.action]
            plot_arrow(state, other_state)


class State2:  # state for infinite horizon problems
    def __init__(self, id, index):
        self.id = id
        self.index = index
        self.neighbours = []
        self.value = 0
        self.next_value = 0
        self.action = 0
        self.next_action = 0

    def __str__(self):
        return 'state: ' + str(self.index)

    def update_value(self):
        difference = np.abs(self.value - self.next_value)  # check convergence
        self.value = self.next_value  # update value
        return difference

    def update_action(self):
        no_change = self.action == self.next_action  # check convergence
        self.action = self.next_action  # update policy
        return no_change


class Reward:       # shortest path rewards
    def __init__(self, states, reward_staying, reward_moving, reward_target):
        self.n = len(states)
        self.matrix = np.full((self.n, self.n), reward_moving)
        self.deterministic = np.full((self.n, self.n), True)
        self.to_r1 = []  # indices of transitions going to r1
        self.to_r2 = []  # indices of transitions going to r2

        for i in range(self.n):
            self.matrix[i, i] = reward_staying

        for ne in states.target.neighbours:
            self.matrix[ne.id, states.target.id] = reward_target

        for nr1 in states.r1.neighbours:
            self.to_r1.append((nr1.id, states.r1.id))
            self.deterministic[nr1.id, states.r1.id] = False

        for nr2 in states.r2.neighbours:
            self.to_r2.append((nr2.id, states.r2.id))
            self.deterministic[nr2.id, states.r2.id] = False

    def __getitem__(self, indices):
        if self.deterministic[indices]:
            return self.matrix[indices]
        else:
            if indices in self.to_r1:
                return choice([-1, -7], 1, p=[0.5, 0.5])[0]
            elif indices in self.to_r2:
                return choice([-1, -2], 1, p=[0.5, 0.5])[0]

    def __str__(self):
        return str(self.matrix)


class Reward2:      # plucking berries
    def __init__(self, states, weights):
        self.n = len(states)
        self.matrix = np.full((self.n, self.n), 0, dtype=np.float)

        for state in states:
            for ne in state.neighbours:
                if ne != state:
                    self.matrix[ne.id, state.id] = weights[state.index]

    def __getitem__(self, indices):
        return self.matrix[indices]

    def __str__(self):
        return str(self.matrix)
