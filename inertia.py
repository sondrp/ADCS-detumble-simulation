import numpy as np

# Inertia of satellite
I = np.array([[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.3]])

invI = np.linalg.inv(I)

LMN_magnetorquer = np.array([0, 0, 0])

k = 67200

n = 84
A = 0.02


def omega_derivative(omega):

    #current = get_current(omega, B)
    #mu_B = current*n*A

    #LMN_magnetorquer = np.cross(mu_B, BB)

    H = I @ omega
    return invI @ (LMN_magnetorquer - np.cross(omega, H))


def get_current(omega, B):
    current = k * np.cross(omega, B) / (n * A)

    if np.sum(np.abs(current)) > 0.04:
        current = current / np.linalg.norm(current) * 0.04

    return current
