import numpy as np

def f(x, t):
    return - (x ** 4) + np.cos(t)


a = 0.0
b = 2
N = 10
h = (b - a) / N

tpoints = np.arange(a, b, h)
xpoints = []

x = 0.0

for t in tpoints:
    xpoints.append(x)
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
    x += k2

print(xpoints)



a = np.arange(0, 10)

b = np.arange(0, 10)

ab = np.subtract.outer(a, b)

#c = np.ones((N, 2)) * np.array([-Lx, Ly])

print(r'Normalized Wave function $\frac{\psi}{\sqrt{|\psi|^2}}$')
