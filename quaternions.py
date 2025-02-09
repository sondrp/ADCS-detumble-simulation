import numpy as np

def euler_to_quaternion(rpy):
    """
    Convert Euler angles to a quaternion.
    
    Parameters:
      rpy: (array)
        roll  (float): x-axis
        pitch (float): y-axis
        yaw   (float): z-axis
    
    Returns:
        np.array: Quaternion [q0, q1, q2, q3]
    """
    roll, pitch, yaw = rpy

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q3 = cr * cp * cy + sr * sp * sy
    q0 = sr * cp * cy - cr * sp * sy
    q1 = cr * sp * cy + sr * cp * sy
    q2 = cr * cp * sy - sr * sp * cy

    return np.array([q0, q1, q2, q3])

def quaternion_derivative(q0123, omega):
    """
    Compute the time derivative of a quaternion given an angular velocity.

    Parameters:
        q (np.array): Quaternion [q0, q1, q2, q3]
        omega (np.array): Angular velocity [ωx, ωy, ωz] (in radians per second)

    Returns:
        np.array: Quaternion derivative [q̇x, q̇y, q̇z, q̇w]
    """
    q0, q1, q2, q3 = q0123
    wx, wy, wz = omega

    return 0.5 * np.array([
        q3 * wx + q1 * wz - q2 * wy,
        q3 * wy + q2 * wx - q0 * wz,
        q3 * wz + q0 * wy - q1 * wx,
       -q0 * wx - q1 * wy - q2 * wz
    ])

def quaternion_to_euler(q0123):
  x, y, z, w = q0123
  # Roll (x-axis rotation)
  sinr_cosp = 2 * (w * x + y * z)
  cosr_cosp = 1 - 2 * (x * x + y * y)
  roll = np.atan2(sinr_cosp, cosr_cosp)

  # Pitch (y-axis rotation)
  sinp = 2 * (w * y - z * x)
  if abs(sinp) >= 1:
      pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
  else:
      pitch = np.asin(sinp)

  # Yaw (z-axis rotation)
  siny_cosp = 2 * (w * z + x * y)
  cosy_cosp = 1 - 2 * (y * y + z * z)
  yaw = np.atan2(siny_cosp, cosy_cosp)

  return roll, pitch, yaw

# TODO : move this somewhere else

# Inertia of satellite
I = np.array([
    [0.9, 0, 0],
    [0, 0.9, 0], 
    [0, 0, 0.3]
])

invI = np.linalg.inv(I)

LMN_magnetorquer = np.array([0, 0, 0])


def omega_derivative(omega):
    H = I @ omega
    return invI @ (LMN_magnetorquer - np.cross(omega, H))