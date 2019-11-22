import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import matplotlib


def run_game(states, ax):
    grid = np.ones(states.grid_size)
    grid[states.bank_position] = 0
    colormap = matplotlib.cm.get_cmap('RdBu')
    ax.matshow(grid, cmap=colormap)
    state = states.initial_state
    (r2, r1) = state.robber.position
    (p2, p1) = state.police.position
    robber_mark, = ax.plot(r1, r2, color=colormap(0.15), marker='$R$', markersize=16)     # create mark for robber
    police_mark, = ax.plot(p1, p2, color=colormap(0.85), marker='$P$', markersize=16)     # create mark for police
    plt.pause(0.5)                          # pause and draw
    while not state.robber_caught():
        state = choice(states.possible_next_states(state=state, action=state.robber.action))
        (r2, r1) = state.robber.position
        (p2, p1) = state.police.position
        robber_mark.set_data(r1, r2)
        police_mark.set_data(p1, p2)
        plt.pause(0.5)  # pause and update positions

