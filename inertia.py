import numpy as np

# Inertia of satellite
I = np.array([[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.3]])

invI = np.linalg.inv(I)

LMN_magnetorquer = np.array([0, 0, 0])

k = k = 0.01  # Reduce gain and tune experimentally

n = 84
A = 0.02


def omega_derivative(omega, B):
    m = (-k / np.linalg.norm(B) ** 2) * np.cross(B, omega)  # magnetic dipole moment

    LMN_magnetorquer = np.cross(m, B)  # control torque

    H = I @ omega  # angular momentum

    # controlled change in angular momentum
    return invI @ (LMN_magnetorquer - np.cross(omega, H))
