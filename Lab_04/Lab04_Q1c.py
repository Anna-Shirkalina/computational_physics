"""
Lab3 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q1b
"""
import numpy as np
import matplotlib.pyplot as plt
from SolveLinear import GaussElim, PartialPivot


def Voltage(x, t, omega): 
    """ Function for voltage with angular frequency as a function of time. 
    Input: time array, angular frequency omega, x as per handout
    Output: V array 
    """
    return x * np.exp(1j * omega * t)


#Constants
R1 = 1e3 # ohms
R3 = 1e3 
R5 = 1e3

R2 = 2e3 # kohms
R4 = 2e3
R6 = 2e3


C1 = 1e-6 
C2 = 0.5e-6

xplus = 3 # volt

omega = 1000 # s^{-1}


#Populate Matrix A
A = np.zeros([3,3], dtype=complex)
A[0,0] = 1/R1 + 1/R4 + omega * 1.0j * C1
A[0,1] = -1j* omega * C1
A[1,0] = -1j * omega* C1
A[1,1] = 1/R2 + 1/R5 + 1j* omega* C1 + 1j * omega * C2
A[1,2] = -1j* omega * C2 
A[2,1] = - 1j * omega * C2 
A[2,2] = 1/R3 + 1/R6 + 1j* omega * C2

# Populate v column
v = np.zeros(3, dtype=complex)
v[0] = xplus/R1
v[1] = xplus/R2
v[2] = xplus/R3  


# find x 
x= PartialPivot(A,v)


print("With no inductor,")

#print absolute values
print("|V_1| is " + str(abs(x[0])) + " V")
print("|V_2| is " + str(abs(x[1])) + " V")
print("|V_3| is " + str(abs(x[2])) + " V")

#print phase at t = 0
print("At t = 0,")
print("V_1 phase is, " + str(np.angle(x[0], deg=True)) +" degrees")
print("V_2 phase is, " + str(np.angle(x[1], deg=True)) +" degrees")
print("V_3 phase is, " + str(np.angle(x[2], deg=True)) +" degrees")


#We plot this for 1 to 2 periods
time = (2 * np.pi)/(omega) * 2

#create time array for 100 sample times
time_array = np.linspace(0, time, num=100)

V1 = Voltage(x[0], time_array, omega)
V2 = Voltage(x[1], time_array, omega)
V3 = Voltage(x[2], time_array, omega)

plt.figure(figsize=(10,5))
plt.plot(time_array, np.real(V1), label="$V_1$")
plt.plot(time_array, np.real(V2), label="$V_2$")
plt.plot(time_array, np.real(V3), label="$V_3$")
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
plt.title('Voltages for a non-inductor circuit')
plt.legend()
plt.savefig('1c_volt.pdf')


#now we do this for a circuit with an inductor. We update the constants
L = R6/omega
R6 = 1j * omega * L

#We update R6 dependendant entries in A
A[2,2] = 1/R3 + 1/R6 + 1j* omega * C2

#We repeat the process

# find x 
x= PartialPivot(A,v)


print("")
print("With an inductor")

#print absolute values
print("|V_1| is " + str(abs(x[0])) + " V")
print("|V_2| is " + str(abs(x[1])) + " V")
print("|V_3| is " + str(abs(x[2])) + " V")

#print phase at t = 0
print("At t = 0,")
print("V_1 phase is, " + str(np.angle(x[0], deg=True)) +" degrees")
print("V_2 phase is, " + str(np.angle(x[1], deg=True)) +" degrees")
print("V_3 phase is, " + str(np.angle(x[2], deg=True)) +" degrees")

# plotting this 
V1 = Voltage(x[0], time_array, omega)
V2 = Voltage(x[1], time_array, omega)
V3 = Voltage(x[2], time_array, omega)

plt.figure(figsize=(10,5))
plt.plot(time_array, np.real(V1), label="$V_1$")
plt.plot(time_array, np.real(V2), label="$V_2$")
plt.plot(time_array, np.real(V3), label="$V_3$")
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
plt.title('Voltages with inductors')
plt.legend()
plt.savefig('1c_volt_inductor.pdf')




