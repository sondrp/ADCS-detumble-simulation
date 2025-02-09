import numpy as np
import matplotlib.pyplot as plt

from igrf import evaluate_magnetic_field_inertial, evaluate_magnetic_field_inertial_quaternion
from orbital_elements import OrbitalElements
from quaternions import euler_to_quaternion, omega_derivative, quaternion_derivative, quaternion_to_euler

### Orbital dynamics ###
# Analytical solution exist, so no need to integrate

orbital_elements = OrbitalElements(6978000, 0, 0, 56 * np.pi / 180, 0, 0)

h = 100
t_final = 5400
steps = int(t_final / h)

t_orbit = np.linspace(0, t_final, steps)
x_orbit = np.zeros((steps, 3))
v_orbit = np.zeros((steps, 3))
mag_orbit = np.zeros((steps, 3))

for i, t in enumerate(t_orbit):
    x, v = orbital_elements.evaluate(t)
    mag = evaluate_magnetic_field_inertial(x)

    x_orbit[i] = x
    v_orbit[i] = v
    mag_orbit[i] = mag

### Rotational dynamics ###
# Integrated with RK4

def derivative(state):
    omega = state[0:3]
    q0123 = state[3:7]
    
    omegadot = omega_derivative(omega)
    q0123dot = quaternion_derivative(q0123, omega)

    return np.hstack([omegadot, q0123dot])

# initial conditions
q0123 = euler_to_quaternion([0, 0, 0])
omega = np.array([0.6, -0.5, -0.4]) # initial angular velocity 

h = 1
t_final = 5400
steps = int(t_final / h)
t_integrated = np.linspace(0, t_final, steps)
omega_integrated = np.zeros((steps, 3))
q0123_integrated = np.zeros((steps, 4))

mag_integrated = np.zeros((steps, 3))

state = np.hstack([omega, q0123])

for i, t in enumerate(t_integrated): 
    q0123 = np.array(state[3:7])
    omega_integrated[i] = np.array(state[0:3])
    q0123_integrated[i] = q0123

    x, _ = orbital_elements.evaluate(t)
    mag_integrated[i] = evaluate_magnetic_field_inertial_quaternion(x, q0123)

    k1 = derivative(state)
    k2 = derivative(state + 0.5*h*k1)
    k3 = derivative(state + 0.5*h*k2)
    k4 = derivative(state + h*k3)

    state += (h/6) * (k1 + 2*k2 + 2*k3 + k4)

### Plotting ###
fig3d = plt.figure(figsize=(6, 5))
ax3d = fig3d.add_subplot(111, projection="3d")
ax3d.scatter(x_orbit[:, 0], x_orbit[:, 1], x_orbit[:, 2], c="b", marker="o")
ax3d.set_title("Orbit plot")

fig, axs = plt.subplots(1, 5, figsize=(12, 5))
ax1, ax2, ax3, ax4, ax5 = axs[0:5]

vnorm = [np.linalg.norm(v) for v in v_orbit]
ax1.plot(t_orbit, v_orbit[:, 0])
ax1.plot(t_orbit, v_orbit[:, 1])
ax1.plot(t_orbit, v_orbit[:, 2])
ax1.plot(t_orbit, vnorm, label="norm")
ax1.set_title("Velocity over time")

magnorm = [np.linalg.norm(mag) for mag in mag_orbit]
ax2.plot(t_orbit, mag_orbit[:, 0])
ax2.plot(t_orbit, mag_orbit[:, 1])
ax2.plot(t_orbit, mag_orbit[:, 2])
ax2.plot(t_orbit, magnorm, label="norm")
ax2.set_title("Magnetic field over time (ECI)")
ax2.legend()

omeganorm = [np.linalg.norm(omega) for omega in omega_integrated]
ax3.plot(t_integrated, omega_integrated[:, 0])
ax3.plot(t_integrated, omega_integrated[:, 1])
ax3.plot(t_integrated, omega_integrated[:, 2])
ax3.plot(t_integrated, omeganorm, label="norm")
ax3.set_title("Angular velocity over time")
ax3.legend()

angles = np.array([quaternion_to_euler(q0123) for q0123 in q0123_integrated])
ax4.plot(t_integrated, angles[:, 0])
ax4.plot(t_integrated, angles[:, 1])
ax4.plot(t_integrated, angles[:, 2])
ax4.set_title("Angles over time")
ax4.legend()

magnorm = [np.linalg.norm(mag) for mag in mag_integrated]
ax5.plot(t_integrated, mag_integrated[:, 0])
ax5.plot(t_integrated, mag_integrated[:, 1])
ax5.plot(t_integrated, mag_integrated[:, 2])
ax5.plot(t_integrated, magnorm, label="norm")
ax5.set_title("Magnetic field over time (body frame)")
ax5.legend()

plt.tight_layout()
plt.show()