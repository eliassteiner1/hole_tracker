#!/usr/bin/env python3
import numpy as np
from   sympy import symbols, Matrix, sin, cos, lambdify


def get_T_tof2imu():
    """
    calculates and returns the transformation matrices from the tof cam to the drone imu and the inverse. everything in meters!
    
    Returns: (tuple)
        T_tof2imu: np.ndarray
        T_imu2tof: np.ndarray
    
    """
    
    INV = lambda x: np.linalg.inv(x)
    
    T_Tof_to_1Tof = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ])

    a = np.pi
    T_1Tof_to_2Tof = np.array([
        [1,         0,          0, 0],
        [0, np.cos(a), -np.sin(a), 0],
        [0, np.sin(a),  np.cos(a), 0],
        [0,         0,          0, 1],
    ])

    b = np.pi / 3
    T_2Tof_to_3Tof = np.array([
        [np.cos(b), -np.sin(b), 0, 0],
        [np.sin(b),  np.cos(b), 0, 0],
        [        0,          0, 1, 0],
        [        0,          0, 0, 1],
    ])

    T_3Tof_to_Drone = np.array([
        [1, 0, 0,  0.1296         ],
        [0, 1, 0, -0.0658         ],
        [0, 0, 1,  0.0960 + 0.0300], # changed ToF mount height by 3cm
        [0, 0, 0,  1.0000         ],
    ])

    T_tof2imu = T_3Tof_to_Drone @ T_2Tof_to_3Tof @ T_1Tof_to_2Tof @ T_Tof_to_1Tof
    T_imu2tof = INV(T_Tof_to_1Tof) @ INV(T_1Tof_to_2Tof) @ INV(T_2Tof_to_3Tof) @ INV(T_3Tof_to_Drone)

    return T_tof2imu, T_imu2tof
    
def get_T_cam2imu():
    """
    calculates and returns the transformation matrices from the rgb cam to the drone imu and the inverse. since the arm can rotate, the function returns lambda. When the arm angle theta is input, the actualy transformation matrix is returned from the lambda. everything in meters!

    Returns: (tuple)
        T_cam2imu: Callable
        T_imu2cam: Callable
    """
    
    T_Cam_to_1 = Matrix([
        [1, 0, 0,  0.0000],
        [0, 1, 0, -0.0600],
        [0, 0, 1,  0.0000],
        [0, 0, 0,  1.0000],
    ])

    th = symbols("th")
    T_1_to_2 = Matrix([
        [cos(th), -sin(th), 0, 0],
        [sin(th),  cos(th), 0, 0],
        [      0,        0, 1, 0],
        [      0,        0, 0, 1],
    ])

    a = -1.5665
    T_2_to_3 = Matrix([
        [1,         0,          0, 0],
        [0, np.cos(a), -np.sin(a), 0],
        [0, np.sin(a),  np.cos(a), 0],
        [0,         0,          0, 1],
    ])

    b = -2.0946
    T_3_to_4 = Matrix([
        [np.cos(b), -np.sin(b), 0, 0],
        [np.sin(b),  np.cos(b), 0, 0],
        [        0,          0, 1, 0],
        [        0,          0, 0, 1],
    ])

    T_4_to_Drone = Matrix([
        [1, 0, 0,  0.3355],
        [0, 1, 0, -0.1699],
        [0, 0, 1,  0.0253],
        [0, 0, 0,  1.0000],
    ])

    T_cam2imu = T_4_to_Drone * T_3_to_4 * T_2_to_3 * T_1_to_2 * T_Cam_to_1
    T_imu2cam = T_Cam_to_1.inv() * T_1_to_2.inv() * T_2_to_3.inv() * T_3_to_4.inv() * T_4_to_Drone.inv()
    
    return lambdify(th, T_cam2imu, "numpy"), lambdify(th, T_imu2cam, "numpy")

