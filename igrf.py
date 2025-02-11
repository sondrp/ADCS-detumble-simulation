import numpy as np
import pyIGRF
from datetime import datetime

R = 6378000  # m, earth radius


def evaluate_magnetic_field_inertial_quaternion(x, q0123):
    """
    The B-dot controller needs magnetic field in body frame, so transform the IGRF result
    with the following function.
    Parameters:
      q0123 array[4]:  quaternion description of the satellite orientation
      x array[4]: position of satellite in ECI frame

    Returns:
      Magnetic field strength in body frame
    """
    q0, q1, q2, q3 = q0123
    BECI = mag_from_pos(x)

    R = np.array(
        [
            [1 - 2 * (q1**2 + q2**2), 2 * (q0 * q1 - q3 * q2), 2 * (q0 * q2 + q3 * q1)],
            [2 * (q0 * q1 + q3 * q2), 1 - 2 * (q0**2 + q2**2), 2 * (q1 * q2 - q3 * q0)],
            [2 * (q0 * q2 - q3 * q1), 2 * (q1 * q2 + q3 * q0), 1 - 2 * (q0**2 + q1**2)],
        ]
    )

    return R @ BECI


# This needs to be double checked
def evaluate_magnetic_field_inertial(x):
    return mag_from_pos(x)


def mag_from_pos(x, time=datetime(2025, 1, 1)):
    lat, long, alt_km = cartesian_to_lat_long(x)
    field = get_magnetic_field(lat, long, alt_km, time)

    return np.array(field[3:6])


def cartesian_to_angles(x):
    x1, x2, x3 = x
    rho = np.linalg.norm(x)
    phi = 0
    theta = np.acos(x3 / rho)
    psi = np.atan2(x2, x1)

    return rho, phi, theta, psi


def cartesian_to_lat_long(x):
    rho, _phi, theta, psi = cartesian_to_angles(x)
    lat = 90 - theta * 180 / np.pi
    long = psi * 180 / np.pi
    alt_km = (rho - R) / 1000

    return lat, long, alt_km


def get_magnetic_field(latitude, longitude, altitude_km, date):
    # Convert date to decimal year
    decimal_year = date.year + (date.timetuple().tm_yday - 1) / 365.25

    # Calculate magnetic field components
    # Returns: [D, I, H, X, Y, Z, F]
    # D: declination (+ve east)
    # I: inclination (+ve down)
    # H: horizontal intensity
    # X: north component
    # Y: east component
    # Z: vertical component (+ve down)
    # F: total intensity
    return pyIGRF.igrf_value(latitude, longitude, altitude_km, decimal_year)


def tib(phi, theta, psi):
    ct = np.cos(theta)
    st = np.sin(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    sy = np.sin(psi)
    cy = np.cos(psi)

    return np.array(
        [
            [ct * cy, sp * st * cy - cp * sy, cp * st * cy + sp * sy],
            [ct * sy, sp * st * sy + cp * cy, cp * st * sy - sp * cy],
            [-st, sp * ct, cp * ct],
        ]
    )
