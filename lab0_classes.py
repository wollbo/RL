from numpy.random import choice
import numpy as np

class State:  # state for finite horizon problems
    time_horizon = None
    start = None
    target = None
    r1 = None
    r2 = None

    def __init__(self, index):
        self.index = index
        self.neighbours = []
        self.value = np.zeros(self.time_horizon)
        self.action = np.zeros(self.time_horizon, dtype=np.uint8)  # local index of best neighbour to go to at time t

    def __str__(self):
        return str(self.index)


class State2:  # state for infinite horizon problems
    precision = None  # epsilon
    discount = None  # lambda
    start = None
    target = None
    r1 = None
    r2 = None

    def __init__(self, index):
        self.index = index
        self.neighbours = []
        self.value = 0
        self.next_value = 0
        self.action = 0
        self.next_action = 0

    def __str__(self):
        return 'state: ' + str(self.index)

    def value_tolerance(self):  # delta
        return self.precision * (1 - self.discount) / self.discount

    def update_value(self):
        no_change = np.abs(self.value - self.next_value) <= self.value_tolerance()  # check convergence
        self.value = self.next_value  # update value
        return no_change

    def update_action(self):
        no_change = self.action == self.next_action  # check convergence
        self.action = self.next_action  # update policy
        return no_change


class Reward:
    def __init__(self, states, target, reward_staying, reward_moving, reward_target, r1, r2):
        self.n = len(states)
        self.matrix = np.full((self.n, self.n), reward_moving)
        self.deterministic = np.full((self.n, self.n), True)
        self.states = states
        self.to_r1 = []     # indices of transitions going to r1
        self.to_r2 = []     # indices of transitions going to r2

        for i in range(self.n):
            self.matrix[i, i] = reward_staying

        for ne in target.neighbours:
            self.matrix[states.index(ne), states.index(target)] = reward_target

        for nr1 in r1.neighbours:
            self.to_r1.append((states.index(nr1), states.index(r1)))
            self.deterministic[states.index(nr1), states.index(r1)] = False

        for nr2 in r2.neighbours:
            self.to_r2.append((states.index(nr2), states.index(r2)))
            self.deterministic[states.index(nr2), states.index(r2)] = False

    def __getitem__(self, indices):
        if self.deterministic[indices]:
            return self.matrix[indices]
        else:
            if indices in self.to_r1:
                # rr = choice([-1, -7], 1, p=[0.5, 0.5])[0]
                # print('Random reward at r1:')
                # print(rr)
                return choice([-1, -7], 1, p=[0.5, 0.5])[0]
            elif indices in self.to_r2:
                return choice([-1, -2], 1, p=[0.5, 0.5])[0]

