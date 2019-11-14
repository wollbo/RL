from numpy.random import choice
import numpy as np
from itertools import count
from lab0_functions import plot_arrow


class StateSpace:       # state space for finite horizon problems
    def __init__(self, maze, time_horizon,
                 start=(None, None), target=(None, None)):
        (n_rows, n_cols) = maze.shape
        self.maze = maze
        self.maze_shape = (n_rows, n_cols)
        self.time_horizon = time_horizon

        self.player_state_count = count(0)
        self.player_states = []
        self.generate_player_states()

        self.minotaur_state_count = count(0)
        self.minotaur_states = []
        self.generate_minotaur_states()

        self.state_count = count(0)
        self.states = []
        self.generate_states()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index]

    def generate_states(self):
        for player_state in self.player_states:
            for minotaur_state in self.minotaur_states:
                state = State(id=next(self.state_count),
                              player=player_state,
                              minotaur=minotaur_state,
                              time_horizon=self.time_horizon)
                self.states.append(state)

    def generate_player_states(self):
        (n_rows, n_cols) = self.maze_shape
        for row in range(n_rows):
            for col in range(n_cols):
                if self.maze[row, col]:
                    state = Player(id=next(self.player_state_count),
                                   position=(row, col),
                                   time_horizon=self.time_horizon)
                    self.player_states.append(state)

        for state in self.player_states:
            state.neighbours.append(state)  # player can stand still
            for other_state in self.player_states:
                if np.abs(other_state.position[0] - state.position[0]) == 1 and np.abs(
                        other_state.position[1] - state.position[1]) == 0:
                    state.neighbours.append(other_state)  # same row neighbours
                elif np.abs(other_state.position[0] - state.position[0]) == 0 and np.abs(
                        other_state.position[1] - state.position[1]) == 1:
                    state.neighbours.append(other_state)  # same column neighbours

    def generate_minotaur_states(self):
        (n_rows, n_cols) = self.maze_shape
        for row in range(n_rows):
            for col in range(n_cols):
                state = Minotaur(id=next(self.minotaur_state_count),
                                      position=(row, col))
                self.minotaur_states.append(state)

        for state in self.minotaur_states:
            # state.neighbours.append(state)  # minotaur can not stand still
            for other_state in self.minotaur_states:
                if np.abs(other_state.position[0] - state.position[0]) == 1 and np.abs(
                        other_state.position[1] - state.position[1]) == 0:
                    state.neighbours.append(other_state)  # same row neighbours
                elif np.abs(other_state.position[0] - state.position[0]) == 0 and np.abs(
                        other_state.position[1] - state.position[1]) == 1:
                    state.neighbours.append(other_state)  # same column neighbours


class Player:
    def __init__(self, id, position, time_horizon):
        self.id = id
        self.position = position
        self.neighbours = []
        self.action = np.zeros(time_horizon, dtype=np.uint8)  # local index of best neighbour to go to at time t


class Minotaur:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.neighbours = []


class State:  # state for finite horizon problems
    def __init__(self, id, player, minotaur, time_horizon):
        self.id = id
        self.player = player
        self.minotaur = minotaur
        self.neighbours = []
        self.value = np.zeros(time_horizon)


    def __str__(self):
        return str(self.index)


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

