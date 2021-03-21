from random import random, randrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import tqdm
import pickle as pickle

T_f = 0.5  # temperature for Monte Carlo
dT = 0.5
T_i = 10
T_steps = int((T_i - T_f) / dT)
T_array = np.arange(T_i, T_f, -dT)
energy_array = pickle.load(open("q3d_energy_per_temp_array.pickle", "rb"))

fig, axes = plt.subplots()
#plt.title('$T$ = {0:.1f}, $N$ = {1:d}'.format(T_f, N))
plt.errorbar(T_array, energy_array[:, 0], yerr=energy_array[:, 1])
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.title("Energy dependance on temperature")
axes.invert_xaxis()
plt.show()
