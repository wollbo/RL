from numpy.random import choice
import numpy as np
from itertools import count
import copy
from lab0_functions import plot_arrow


class StateSpace:       # state space for finite horizon problems
    def __init__(self, maze, time_horizon, exit):
        (n_rows, n_cols) = maze.shape
        self.maze = maze
        self.initial_state = None
        self.exit = exit
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
        self.minotaur_states_for_given_player_state = []
        self.generate_states()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index]

    def values(self, t):
        values = np.empty(len(self.states))
        for s in self.states:
            values[s.id] = s.value[t]
        return values

    def possible_next_states(self, state, action):  # j \in S

        if state.minotaur.position == state.player.position:
            possible_next_states = [state]

        elif state.player.position == self.exit:
            possible_next_states = [state]

        else:
            next_player_state = state.player.neighbours[action]
            possible_subset = self.minotaur_states_for_given_player_state[next_player_state.id]
            current_minotaur = state.minotaur
            possible_next_states = []

            for next_state in possible_subset:
                for neighbouring_minotaur in current_minotaur.neighbours:
                    if next_state.minotaur.position == neighbouring_minotaur.position:
                        possible_next_states.append(next_state)

        return possible_next_states

    def generate_states(self):
        for player_state in self.player_states:
            minotaur_states_for_this_player_state = []
            for minotaur_state in self.minotaur_states:
                state = State(id=next(self.state_count),
                              player=copy.deepcopy(player_state),
                              minotaur=copy.deepcopy(minotaur_state),
                              time_horizon=self.time_horizon)

                if state.player.position == (0, 0) and state.minotaur.position == (6, 5):
                    self.initial_state = state

                self.states.append(state)

                minotaur_states_for_this_player_state.append(state)
            self.minotaur_states_for_given_player_state.append(minotaur_states_for_this_player_state)


    def generate_player_states(self):
        (n_rows, n_cols) = self.maze_shape
        for row in range(n_rows):
            for col in range(n_cols):
                if self.maze[row, col]:
                    state = Player(id=next(self.player_state_count),
                                   position=(row, col),
                                   time_horizon=self.time_horizon,
                                   exit=self.exit)
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


class State:  # state for finite horizon problems
    def __init__(self, id, player, minotaur, time_horizon):
        self.id = id
        self.player = player
        self.minotaur = minotaur
        self.neighbours = []
        self.value = np.zeros(time_horizon)

        self.losing = False     # TODO
        self.winning = False    # TODO

    def __str__(self):
        return str(self.player) + '\n' + str(self.minotaur)


class Player:
    def __init__(self, id, position, time_horizon, exit):
        self.id = id
        self.position = position
        self.neighbours = []
        self.action = np.zeros(time_horizon, dtype=np.uint8)  # local index of best neighbour to go to at time t
        self.exit = exit

    def __str__(self):
        return 'player @ ' + str(self.position)


class Minotaur:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.neighbours = []

    def __str__(self):
        return 'minotaur @ ' + str(self.position)


class Reward:
    def __init__(self, reward_eaten, reward_exiting, reward_staying=0, reward_moving=0):
        self.reward_eaten = reward_eaten
        self.reward_exiting = reward_exiting
        self.reward_staying = reward_staying
        self.reward_moving = reward_moving

    def reward(self, player, minotaur):
        if player.position == minotaur.position:
            return self.reward_eaten

        elif player.position == player.exit:
            return self.reward_exiting

        else:
            return 0
