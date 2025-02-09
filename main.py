import numpy as np
import matplotlib.pyplot as plt

from igrf import evaluate_magnetic_field_inertial
from orbital_elements import OrbitalElements

a = 400000+6378000  # meters, ISS height + earth radius
e = 0  # the ISS has very close to 0 eccentricity

i = 56 * np.pi / 180  # inclination of ISS
raan = 0.0
aop = 0.0
m = 0.0

# maybe rather calculate the time it takes to go around once?

t_final = 5400
h = 100
steps = int(t_final / h)

tout = np.linspace(0, t_final, steps)
xout = np.zeros((steps, 3))
vout = np.zeros((steps, 3))
magout = np.zeros((steps, 3))

orbital_elements = OrbitalElements(a, e, m, i, raan, aop)

for i, t in enumerate(tout):
    x, v = orbital_elements.evaluate(t)
    mag = evaluate_magnetic_field_inertial(x)

    xout[i] = x
    vout[i] = v
    magout[i] = mag

# orbit
fig3d = plt.figure(figsize=(6, 5))
ax3d = fig3d.add_subplot(111, projection="3d")
ax3d.scatter(xout[:, 0], xout[:, 1], xout[:, 2], c="b", marker="o")
ax3d.set_title("Orbit plot")
plt.tight_layout()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax1, ax2 = axs[0:2]

vnorm = [np.linalg.norm(v) for v in vout]
ax1.plot(tout, vout[:, 0])
ax1.plot(tout, vout[:, 1])
ax1.plot(tout, vout[:, 2])
ax1.plot(tout, vnorm, label="norm")
ax1.set_title("Velocity over time")


magnorm = [np.linalg.norm(mag) for mag in magout]
ax2.plot(tout, magout[:, 0])
ax2.plot(tout, magout[:, 1])
ax2.plot(tout, magout[:, 2])
ax2.plot(tout, magnorm, label="norm")
ax2.set_title("Magnetic field over time")
ax2.legend()

plt.tight_layout()
plt.show()