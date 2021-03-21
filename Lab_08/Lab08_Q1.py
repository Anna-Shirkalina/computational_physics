import numpy as np
import matplotlib.pyplot as plt
from time import time

# Constants
M = 100  # Grid squares on a side
V_1 = 1.0  # Voltage for first plate
V_2 = -1.0  # voltage for the second plate
ppc = 10 / 100  # number of points per centimeter
x_plate1 = int(2 / ppc)
x_plate2 = int(8 / ppc)
y_bottom = int(2 / ppc)
y_top = int(8 / ppc)
target = 1e-6 # [V] Target accuracy
# Create arrays to hold potential values
phi = np.zeros([M+1, M+1], float)
phi[int(2/ppc):int(8/ppc) + 1, x_plate1] = V_1
phi[int(2/ppc):int(8/ppc) + 1, x_plate2] = V_2


def guass_seidel(phi, x1, x2, yt, yb, w=0.0):
    """Use the Guass-Seidel method as provided in Newman to calculate the value
    of a PDE at each point, this function is taylored to the 2 plate method
    x1 and x2 refers to the x index of plate 1 and 2 respectively
    yt and yb refers to the y index of top and bottom of the plates
    """
    # Main loop
    delta = 1.0
    count = 0
    while delta > target:
        delta = 0.0
        count += 1
        for i in range(1, M):  # CHANGE HERE: boundaries never updated
            for j in range(1, M):  # CHANGE HERE
                if (j == x1 and yb <= i <= yt) or (j == x2 and yb <= i <= yt):
                    continue
                phi_old = phi[i, j]
                phi[i, j] = (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1]
                             + phi[i, j - 1]) * ((1 + w) / 4) - w * phi[i, j]
                # keep track of the maximum error
                phi_new = phi[i, j]
                delta = max(delta, np.abs(phi_new - phi_old))

    return phi, count
omega = 0.5
# Make a plot
a = time()
phi, iter = guass_seidel(phi, x_plate1, x_plate2, y_top, y_bottom, omega)
b = time()
print(f"The while loop took {iter} iterations")
print(f"It took {b - a} seconds to find the solution")
np.savetxt('potential_Q1.csv', phi, delimiter=',')


plt.figure(1)
plt.imshow(phi)
cbar=plt.colorbar()
cbar.set_label('Potential $V$')
plt.title('Solution for electostatic potential $\omega$ =' + str(omega))
plt.xlabel('x in mm')
plt.ylabel('y in mm')
plt.axis('equal')
plt.tight_layout()
plt.gray()
#plt.savefig('Q1_b_i_w_0.5.pdf')
plt.savefig('Q1_a_i.pdf')



fig = plt.figure()
x = np.linspace(0, 10, M + 1)  # Phi actually has the size M + 1
y = np.linspace(0, 10, M + 1)
X, Y = np.meshgrid(x, y)
Ey, Ex = np.gradient(-phi, y, x)
strm = plt.streamplot(X, Y, Ex, Ey, color=phi, linewidth=2, cmap='autumn')
cbar = fig.colorbar(strm.lines)
cbar.set_label('Potential $V$')
plt.title('Electric field lines $\omega$ =' + str(omega))
plt.axis('equal')
plt.xlabel('$x$ (mm)')
plt.ylabel('$y$ (mm)')
plt.tight_layout()
plt.savefig('Q1_b_ii_w_{}.pdf'.format(str(omega)))
plt.savefig('Q1_a_ii.pdf')
plt.show()
