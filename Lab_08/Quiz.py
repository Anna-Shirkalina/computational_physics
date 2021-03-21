import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(-2, 2, 100) # does not include zero
# y = np.linspace(-1, 1, 50)
# X, Y = np.meshgrid(x, y)
# R1 = ((X-1)**2 + Y**2)**.5 # 1st charge located at x=+1, y=0
# R2 = ((X+1)**2 + Y**2)**.5 # 2nd charge located at x=-1, y=0
#
# V = 1./R1 - 1./R2 # two equal-and-opposite charges
# Ey, Ex = np.gradient(-V, y, x) # careful about order
# fig = plt.figure(figsize=(6, 3))
# strm = plt.streamplot(X, Y, Ex, Ey, color=V, linewidth=2, cmap='autumn')
# cbar = fig.colorbar(strm.lines)
# cbar.set_label('Potential $V$')
# plt.title('Electric field lines')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.axis('equal')
# plt.tight_layout()
#
# plt.figure(2)
# fig2 = plt.imshow(V)
# plt.show()

M = 100
phi = np.loadtxt('potential_Q1.csv', delimiter=',')
x = np.linspace(0, 10, M + 1)
y = np.linspace(0, 10, M + 1)
X, Y = np.meshgrid(x, y)
Ey, Ex = np.gradient(phi, y, x)

plt.figure(1)
plt.imshow(phi)
cbar = plt.colorbar()
cbar.set_label('Potential $V$')
plt.title('Solution for electostatic potential')
plt.xlabel('x in cm')
plt.ylabel('y in cm')
plt.gray()


fig = plt.figure(figsize=(6, 3))
strm = plt.streamplot(X, Y, Ex, Ey, color=phi, linewidth=2, cmap='autumn')
cbar = fig.colorbar(strm.lines)
cbar.set_label('Potential $V$')
plt.title('Electric field lines')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
plt.show()

print('done')
