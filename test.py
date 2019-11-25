import numpy as np
import matplotlib.pyplot as plt

t_log = []

for t in range(1, 1000):
    if round(1000 * np.log10(t)) % 1000 == 0:
        print(t)
        print(np.log10(t))
        t_log.append(t)









