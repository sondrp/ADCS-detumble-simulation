import numpy as np
import pyIGRF
from datetime import datetime

R = 6378000  # m


def evaluate_magnetic_field_inertial(x):
    rho, phi, theta, psi = cartesian_to_angles(x)
    lat, long, alt_km = cartesian_to_lat_long(rho, phi, theta, psi)
    field = get_magnetic_field(lat, long, alt_km, datetime(2025, 1, 1))

    BNED = np.array(field[3:6])
    return tib(phi, theta, psi) @ BNED

def cartesian_to_angles(x):
  x1, x2, x3 = x
  rho = np.linalg.norm(x)
  phi = 0
  theta = np.acos(x3 / rho)
  psi = np.atan2(x2, x1)

  return rho, phi, theta, psi
    


def cartesian_to_lat_long(rho, phi, theta, psi):
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

    return np.array([
      [ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
      [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
      [-st, sp*ct, cp*ct]
    ])