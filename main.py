import numpy as np
import matplotlib.pyplot as plt

from orbital_elements import orbital_elements_to_position_and_velocity


def magnetic_moment(n, A, i_M):
  """
    The magnetic moment produced by the magnetorquer is given by
    mu_M = nAi_M , where

    n: number of coils
    A: area of magnetorquer
    i_M is a vector consisting of the magnitude of current through each magnetorquer

    Returns: mu_M  , a vector with magnetic moment in each direction
  """
  return n* A * i_M

def torque(mu_M, B):
  """
  The torque produced by the magnetorquers due to interaction with Earth's magnetic field is given by
  t_M = mu_M x B, where

  mu_M: magnetic moment produced by magnetorquer
  B: vector, earth's magnetic field

  Returns t_M, a vector with torques
  """
  return np.cross(mu_M, B)

def b_dot_controller(k, Bdot):
  """
  B-Dot controller is a control law used to detumble CubeSats given by
  mu_M = -k Bdot, where

  k: control gain
  Bdot: the derivative of magnetic field (in the body frame)

  Returns: vector, magnetic moment command to magnetorquers
  """
  return -k * Bdot

def magnetic_field_body_frame(q0123, B_ECI):
  """
  The B-dot controller needs magnetic field in body frame, so transform the IGRF result
  with the following function. 

  q0123: array[4] quaternion description of the satellite orientation
  B_ECI: magnetic field in the earth center frame (result from IGRF model)

  Returns magnetic field strength in body frame
  """
  q0, q1, q2, q3 = q0123

  A = np.array([
    [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
    [2*(q1*q2 - q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
    [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3,]
  ])

  return A*B_ECI




a = 1.0  # AU
e = 0.0167

i = 0.0  
raan = 0.0
aop = 0.0
m = 0.0

t = 0
h = 0.1

xs = []
vs = []

while t < 10:
  x, v = orbital_elements_to_position_and_velocity(a, e, m, i, raan, aop, t)
  xs.append(x)
  vs.append(v)

  t += h


# Convert lists to arrays for plotting
xs = np.array(xs)
vs = np.array(vs)

# Create a plot of position (x) vs. time and velocity (v) vs. time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot position over time
ax1.plot(np.arange(0, 10, h), xs)
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Position (x)')
ax1.set_title('Position vs. Time')

# Plot velocity over time
ax2.plot(np.arange(0, 10, h), vs)
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Velocity (v)')
ax2.set_title('Velocity vs. Time')

# Show the plot
plt.tight_layout()
plt.show()



