import numpy as np
import matplotlib.pyplot as plt


# Solve Kepler's equation for E
#   M = E - e sin E
# Transcendental function, must use newton's method
def newton_rhapson(M, e, tol=10e-12):
    E = (
        M
        + e * np.sin(M) / (1 - e * np.cos(M))
        - 0.5 * (e * np.sin(M) / (1 - e * np.cos(M))) ** 3
    )

    delta_E = 1

    while delta_E > tol:
        delta_E = (M - (E - e * np.sin(E))) / (1 - e * np.cos(E))
        E += delta_E

    return E


class OrbitalElements:
    
    # G * M
    mu = 5.9722e24 * 6.6743e-11 

    def __init__(self, a, e, m0, i, raan, aop):
        """
        a   : [float] semimajor axis
        e   : [float] eccentricity
        m0  : [float] initial mean anomaly
        i   : [float] inclination
        raan: [float] right ascension of the ascending node (uppercase Omega)
        aop : [float] argument of perigee                   (lowercase omega)
        """

        A11 = np.cos(raan) * np.cos(aop) - np.sin(raan) * np.sin(aop) * np.cos(i)
        A12 = np.sin(raan) * np.cos(aop) + np.cos(raan) * np.sin(aop) * np.cos(i)
        A13 = np.sin(aop) * np.sin(i)

        A21 = -np.cos(raan) * np.sin(aop) - np.sin(raan) * np.cos(aop) * np.cos(i)
        A22 = -np.sin(raan) * np.sin(aop) + np.cos(raan) * np.cos(aop) * np.cos(i)
        A23 = np.cos(raan) * np.sin(i)

        self.a = a
        self.e = e
        self.m0 = m0

        self.A = np.array(
            [
                [A11, A21],
                [A12, A22],
                [A13, A23],
            ]
        )

    def evaluate(self, t):
        """
        t: time since passage of perigee
        returns: array([x1, x2, x3, v1, v2, v3]) # position and velocity
        """
        a, e, m0 = self.a, self.e, self.m0
        mu, A = self.mu, self.A

        n = np.sqrt(mu / a**3)

        m = m0 + n * t

        E = newton_rhapson(m, e)

        r = a * (1 - e * np.cos(E))
        x = a * (np.cos(E) - e)
        y = a * np.sqrt(1 - e * e) * np.sin(E)

        xdot = -(n * a * a / r) * np.sin(E)
        ydot = (n * a * a / r) * np.sqrt(1 - e * e) * np.cos(E)

        pos = A @ np.array([x, y])
        vel = A @ np.array([xdot, ydot])

        return pos, vel


# Example:

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

orbital_elements = OrbitalElements(a, e, m, i, raan, aop)

for i, t in enumerate(tout):
    x, v = orbital_elements.evaluate(t)
    xout[i] = x
    vout[i] = v

fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")

x = xout[:, 0]
y = xout[:, 1]
z = xout[:, 2]

ax1.scatter(x, y, z, c="b", marker="o")

# Set equal aspect ratio
max_range = (
    np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
)
mid_x = (x.max() + x.min()) / 2.0
mid_y = (y.max() + y.min()) / 2.0
mid_z = (z.max() + z.min()) / 2.0

ax1.set_xlim(mid_x - max_range, mid_x + max_range)
ax1.set_ylim(mid_y - max_range, mid_y + max_range)
ax1.set_zlim(mid_z - max_range, mid_z + max_range)

ax1.set_xlabel("X Axis")
ax1.set_ylabel("Y Axis")
ax1.set_zlabel("Z Axis")

ax1.set_xlabel("X Axis")
ax1.set_ylabel("Y Axis")
ax1.set_zlabel("Z Axis")
ax1.set_title("3D Surface Plot")

plt.tight_layout()
plt.show()

vnorm = [np.linalg.norm(v) for v in vout]
plt.plot(tout, vout[:, 0])
plt.plot(tout, vout[:, 1])
plt.plot(tout, vout[:, 2])
plt.plot(tout, vnorm)
plt.show()