##Starting the code
##Insert the variables

import numpy as np
import matplotlib.pyplot as plt
import torch

# Define the variables

# Example usage
t = np.linspace(0,1,20) # Example time value
Omega = [0.5, 0.7]  # Represents the Rabi frequency or coupling strength between quantum states
tau = 0.1  # Interaction time, controlling how long the quantum system interacts with external fields
delta = [2.0, 3.0]  # Detuning values, representing the difference between the driving field frequency and the system's resonance frequency
phi = [np.pi/4, np.pi/2]  # Phase values, controlling the initial phase of the oscillations

def probability_t(t, Omega, tau, delta, phi):
    return 1/2 + sum(Omega[k] * tau * np.sin(delta[k] * t + phi[k]) for k in range(2))
p_t = probability_t(t, Omega, tau, delta, phi)
print(p_t)

probability_t(t, Omega, tau, delta, phi)



# Plotting the results
plt.plot(t, p_t, label='Probability')
plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.title('Probability vs Time')
plt.legend()
plt.grid()
plt.show()
