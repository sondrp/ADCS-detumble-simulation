import numpy as np



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