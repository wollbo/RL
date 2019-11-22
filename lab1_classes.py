import numpy as np
from itertools import count
import copy


class StateSpace:       # state space for finite horizon problems
    def __init__(self, mdp, maze, entry=(0, 0), exit=(6, 5), discount_factor=0.95, precision=1, time_horizon=20, minotaur_can_stand_still=False):
        (n_rows, n_cols) = maze.shape
        self.maze = maze
        self.entry = entry
        self.exit = exit
        self.initial_state = None   # initialized in generate_states
        self.terminal_state = None  # same
        self.minotaur_can_stand_still = minotaur_can_stand_still
        self.maze_shape = (n_rows, n_cols)

        if mdp == 'discounted':
            self.mdp = mdp
            self.infinite_horizon_discounted = True
            self.discount_factor = discount_factor
            self.precision = precision
            self.tolerance = self.precision * (1 - self.discount_factor) / self.discount_factor
            self.finite_horizon = False

        elif mdp == 'finite':
            self.finite_horizon = True
            self.mdp = mdp
            self.time_horizon = time_horizon
            self.infinite_horizon_discounted = False
        else:
            print('Specify MDP please.')

        self.player_states = []
        self.generate_player_states()
        self.n_player_states = len(self.player_states)

        self.minotaur_states = []
        self.generate_minotaur_states()
        self.n_minotaur_states = len(self.minotaur_states)

        self.states = []
        self.states_matrix = np.full((len(self.player_states), len(self.minotaur_states)), State)
        self.generate_states()

        if mdp == 'discounted':
            self.value_iteration_difference = np.full(len(self), 1, dtype=np.float)

        print('States generated.')

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index]

    def values(self, t=0):
        values = np.empty(len(self.states))
        if self.finite_horizon:
            for s in self.states:
                values[s.id] = s.value[t]

        elif self.infinite_horizon_discounted:
            for s in self.states:
                values[s.id] = s.value
        return values

    def set_initial_values(self, v0):
        if self.infinite_horizon_discounted:
            for state in self.states:
                state.value = v0[state.id]
        else:
            pass

    def subset_where(self, player_id=None, minotaur_id=None):
        if minotaur_id is None and player_id in range(self.n_player_states):
            return list(self.states_matrix[player_id, :])

        elif player_id is None and minotaur_id in range(self.n_minotaur_states):
            return list(self.states_matrix[:, minotaur_id])

        else:
            print('Error: Specify a valid id for a player or a minotaur state.')
            return None

    def possible_next_states(self, state, action):  # j \in S
        if state == self.terminal_state:
            possible_next_states = [state]

        elif state.losing() or state.winning():
            possible_next_states = [self.terminal_state]

        # if state.losing():
        #     possible_next_states = [state]
        #
        # elif state.winning():
        #     possible_next_states = [state]

        else:
            next_player_state = state.player.neighbours[action]
            subset = self.subset_where(player_id=next_player_state.id)
            possible_next_states = [subset[next_minotaur.id] for next_minotaur in state.minotaur.neighbours]

        return possible_next_states

    def update_values(self):
        for state in self.states:
            self.value_iteration_difference[state.id] = state.update_value()

    def stopping_condition(self):
        return all(np.less(self.value_iteration_difference, self.tolerance))

    def generate_states(self):
        state_counter = count(0)
        for player_state in self.player_states:
            for minotaur_state in self.minotaur_states:
                if self.mdp == 'finite':
                    state = State(id=next(state_counter),
                                  mdp=self.mdp,
                                  player=copy.deepcopy(player_state),
                                  minotaur=minotaur_state,   # does not need to be deep copy really ??
                                  time_horizon=self.time_horizon)
                elif self.mdp == 'discounted':
                    state = State(id=next(state_counter),
                                  mdp=self.mdp,
                                  player=copy.deepcopy(player_state),
                                  minotaur=minotaur_state)

                if state.player.position == self.entry and state.minotaur.position == self.exit:
                    self.initial_state = state

                self.states.append(state)
                self.states_matrix[player_state.id, minotaur_state.id] = state

        terminal_player_state = [ps for ps in self.player_states if ps.position == self.entry][0]
        terminal_minotaur_state = [ms for ms in self.minotaur_states if ms.position == self.exit][0]

        if self.mdp == 'finite':
            terminal_state = State(id=next(state_counter),
                                   mdp=self.mdp,
                                   player=copy.deepcopy(terminal_player_state),
                                   minotaur=terminal_minotaur_state,  # does not need to be deep copy really ??
                                   time_horizon=self.time_horizon)
        elif self.mdp == 'discounted':
            terminal_state = State(id=next(state_counter),
                                   mdp=self.mdp,
                                   player=copy.deepcopy(terminal_player_state),
                                   minotaur=copy.deepcopy(terminal_minotaur_state))
        self.states.append(terminal_state)
        self.terminal_state = terminal_state

    def generate_player_states(self):
        (n_rows, n_cols) = self.maze_shape
        player_state_counter = count(0)
        for row in range(n_rows):
            for col in range(n_cols):
                if self.maze[row, col]:
                    if self.mdp == 'finite':
                        state = Player(id=next(player_state_counter),
                                       mdp=self.mdp,
                                       position=(row, col),
                                       time_horizon=self.time_horizon,
                                       exit=self.exit)
                    elif self.mdp == 'discounted':
                        state = Player(id=next(player_state_counter),
                                       mdp=self.mdp,
                                       position=(row, col),
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
        minotaur_state_counter = count(0)
        for row in range(n_rows):
            for col in range(n_cols):
                state = Minotaur(id=next(minotaur_state_counter),
                                 position=(row, col))
                self.minotaur_states.append(state)

        for state in self.minotaur_states:
            if self.minotaur_can_stand_still:   # minotaur can not stand still ?
                state.neighbours.append(state)

            for other_state in self.minotaur_states:
                if np.abs(other_state.position[0] - state.position[0]) == 1 and np.abs(
                        other_state.position[1] - state.position[1]) == 0:
                    state.neighbours.append(other_state)  # same row neighbours
                elif np.abs(other_state.position[0] - state.position[0]) == 0 and np.abs(
                        other_state.position[1] - state.position[1]) == 1:
                    state.neighbours.append(other_state)  # same column neighbours


class State:
    def __init__(self, mdp, id, player, minotaur, time_horizon=20):
        self.id = id
        self.player = player
        self.minotaur = minotaur
        self.neighbours = []

        if mdp == 'finite':
            self.value = np.zeros(time_horizon)
        elif mdp == 'discounted':
            self.value = 0
            self.next_value = 0

    def __str__(self):
        return str(self.player) + '\n' + str(self.minotaur)

    def losing(self):
        if self.minotaur.position == self.player.position:
            return True
        else:
            return False

    def winning(self):
        if self.losing():
            return False
        else:
            return self.player.is_at_exit()

    def update_value(self):
        difference = np.abs(self.value - self.next_value)  # check convergence
        self.value = self.next_value  # update value
        return difference


class Player:
    def __init__(self, id, mdp, position,  exit, time_horizon=20):
        self.id = id
        self.position = position
        self.exit = exit
        self.neighbours = []

        if mdp == 'finite':
            self.action = np.zeros(time_horizon, dtype=np.uint8)  # local index of best neighbour to go to at time t

        elif mdp == 'discounted':
            self.action = 0
            self.next_action = 0

    def __str__(self):
        return 'player @ ' + str(self.position)

    def is_at_exit(self):
        return self.position == self.exit

    def update_action(self):
        no_change = self.action == self.next_action  # check convergence
        self.action = self.next_action  # update policy
        return no_change


class Minotaur:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.neighbours = []

    def __str__(self):
        return 'minotaur @ ' + str(self.position)


class Reward:
    def __init__(self, reward_eaten, reward_exiting):
        self.reward_eaten = reward_eaten
        self.reward_exiting = reward_exiting

    def __call__(self, state):
        if state.winning():
            return self.reward_exiting
        elif state.losing():
            return self.reward_eaten
        else:
            return 0

