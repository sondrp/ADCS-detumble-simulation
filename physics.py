import numpy as np
import matplotlib.pyplot as plt

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

  return A @ B_ECI



