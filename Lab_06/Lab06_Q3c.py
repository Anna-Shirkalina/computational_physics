import numpy as np
import matplotlib.pyplot as plt
from Lab06_Functions import *
from tqdm import tqdm


N = 16
t_stop = 10
dt = 0.01
Lx = 4.0
Ly = 4.0

dx = Lx / np.sqrt(N)
dy = Ly / np.sqrt(N)

x_grid = np.arange(0.5 * dx, Lx, dx)
y_grid = np.arange(0.5 * dy, Ly, dy)

xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

x_initial = xx_grid.flatten()
y_initial = yy_grid.flatten()
position_init = np.stack((x_initial, y_initial), axis=1)

vel_init = np.zeros((N, 2))  # number of particles and 2 dimensions


M = N * 9
index_main = N * 4
time_array = np.arange(0, t_stop / dt, dt)
num_steps = int(t_stop / dt)
velocity = np.zeros((num_steps, M, 2))
velocity_odd = np.zeros((num_steps, M, 2))
positions = np.zeros((num_steps, M, 2))
main_tile = np.zeros((num_steps, N, 2))


main_tile[0] = position_init
positions[0] = build_block(position_init, Lx, Ly)
velocity_odd[0] = velocity[0] + dt * acceleration(positions[0])

for t in tqdm(range(num_steps - 1)):
    positions[t + 1] = positions[t] + dt * velocity_odd[t]         # 144 by 2
    # make sure that the x particles are coming back
    main_tile[t + 1] = positions[t + 1][N * 4: N * 5]               # 16 by 2
    main_tile[t + 1, :, 0] = np.mod(main_tile[t + 1, :, 0], Lx)          # 16 by 2
    main_tile[t + 1, :, 1] = np.mod(main_tile[t + 1, :, 1], Ly)

    k = dt * acceleration(positions[t + 1])  # 144 by 2
    velocity[t + 1] = velocity_odd[t] + 0.5 * k  # 144 by 2
    velocity_odd[t + 1] = velocity_odd[t] + k

    # make tiles recreate 144 by 2
    positions[t + 1] = build_block(main_tile[t + 1], Lx, Ly)
    # extract middle velocity and stack them
    vel_main_tile = velocity[t + 1][N * 4: N * 5]
    b = np.tile(vel_main_tile, (9, 1))
    velocity[t + 1] = np.tile(vel_main_tile, (9, 1))

plt.figure(1)
for particle in range(N):
    name = particle + 1
    plt.plot(main_tile[:, particle, 0], main_tile[:, particle, 1], label=name)

plt.legend(loc='right')
plt.xlim(right=5)
plt.xlabel("x distance")
plt.ylabel("y distance")
plt.show()

