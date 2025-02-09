import numpy as np
import matplotlib.pyplot as plt

M = 5.9722e24
G = 6.6743e-11

gravitational_parameter = G * M  # the greek letter mu in literature

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


class RotationMatrix:

    # We only need the transpose of the first two rows of the 3-1-3 Euler rotation matrix
    def __init__(self, raan=None, aop=None, i=None):
        self.A = None
        self._create_matrix(raan, aop, i)

    def multiply(self, x, y):
        if self.A is None:
            return None

        vector = np.array([x, y])
        return self.A @ vector

    # Make sure the stored matrix is correct
    def prepare(self, raan, aop, i):
        if raan == self.raan and aop == self.aop and i == self.i:
            return

        self._create_matrix(raan, aop, i)

    def _create_matrix(self, raan, aop, i):
        self.raan = raan
        self.aop = aop
        self.i = i

        if raan is None or aop is None or i is None:
            return

        A11 = np.cos(raan) * np.cos(aop) - np.sin(raan) * np.sin(aop) * np.cos(i)
        A12 = np.sin(raan) * np.cos(aop) + np.cos(raan) * np.sin(aop) * np.cos(i)
        A13 = np.sin(aop) * np.sin(i)

        A21 = -np.cos(raan) * np.sin(aop) - np.sin(raan) * np.cos(aop) * np.cos(i)
        A22 = -np.sin(raan) * np.sin(aop) + np.cos(raan) * np.cos(aop) * np.cos(i)
        A23 = np.cos(raan) * np.sin(i)

        self.A = np.array(
            [
                [A11, A21],
                [A12, A22],
                [A13, A23],
            ]
        )


rotation_matrix = RotationMatrix()


def orbital_elements_to_position_and_velocity(a, e, m0, i, raan, aop, t):
    """
    a   : [float] semimajor axis
    e   : [float] eccentricity
    m0  : [float] initial mean anomaly
    i   : [float] inclination
    raan: [float] right ascension of the ascending node (uppercase Omega)
    aop : [float] argument of perigee                   (lowercase omega)
    t   : [float] time since perigee passage

    returns: position and velocity vectors at time t in orbit
    """

    n = np.sqrt(gravitational_parameter / a**3)

    m = m0 + n * t

    E = newton_rhapson(m, e)

    r = a * (1 - e * np.cos(E))
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e * e) * np.sin(E)

    xdot = -(n * a * a / r) * np.sin(E)
    ydot = (n * a * a / r) * np.sqrt(1 - e * e) * np.cos(E)

    rotation_matrix.prepare(raan, aop, i)

    pos = rotation_matrix.multiply(x, y)
    vel = rotation_matrix.multiply(xdot, ydot)

    return pos, vel


# Example:

a = 1.0  # AU
e = 0   # the ISS has very close to 0 inclination

i = 56*np.pi / 180  
raan = 0.0
aop = 0.0
m = 0.0

h = 0.01
steps = int(2 / h)
tout = np.linspace(0, 2, steps)
xout = np.zeros((steps, 3))
vout = np.zeros((steps, 3))

for index, t in enumerate(tout):
  x, v = orbital_elements_to_position_and_velocity(a, e, m, i, raan, aop, t)
  xout[index] = x
  vout[index] = v
 
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x = xout[:, 0]
y = xout[:, 1]
z = xout[:, 2]

ax.scatter(x, y, z, c='b', marker='o')

# Set equal aspect ratio
max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
mid_x = (x.max() + x.min()) / 2.0
mid_y = (y.max() + y.min()) / 2.0
mid_z = (z.max() + z.min()) / 2.0

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.tight_layout()
plt.show()
