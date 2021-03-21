"""
Author: Nicolas Grisouard, Univ. Toronto
# adapted from https://matplotlib.org/3.3.1/api/animation_api.html
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc

font = {'family': 'DejaVu Sans', 'size': 14}  # adjust fonts
rc('font', **font)

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'r--')  # red dashed line

x = np.linspace(0., 4*np.pi, 128)  # x positions array
t = np.linspace(0., 4*np.pi, 128)  # time array

X, T = np.meshgrid(x, t)

data = np.sin(X-T)


def init():  # the FuncAnimation function requires functions as arguments
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('$x$')
    plt.grid()
    return ln,


def to_be_animated(time):
    ax.set_title(r'$\sin(x-t)$ at $t = {0:05.2f}$'.format(time))
    # above: "05.2f" means: string of length 5 (the point counts), 2 digits
    # after the point. If less digits than 5, pad left with zeroes.
    # It is crucial for animations to keep the length of updated labels under
    # control, otherwise the labels could shift positions all the time
    plt.tight_layout()
    ln.set_data(x, data[:, time])
    return ln,


ani = FuncAnimation(fig,  # figure identifyer object
                    to_be_animated,  # the function to animate in time
                    frames=range(0, len(t), 4),  # plotting every four frames
                    init_func=init,
                    interval=100,  # [s]  time between each frame
                    blit=True)  # blit speeds up the execution of script

ani.save('animation.mp4', dpi=100)
