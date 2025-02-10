#!/usr/bin/env python3
import numpy as np


def construct_camera_intrinsics(f_hat_x: float, f_hat_y: float, cx: float, cy: float):
    intrinsics = np.array([
        [f_hat_x,       0,     cx],
        [      0, f_hat_y,     cy],
        [      0,       0,      1],
    ], dtype=np.float32)
    
    return intrinsics

def quat_2_rot(x: float, y: float, z: float, w: float):
    """
    Convert a quaternion to a rotation matrix, standard implementation
    """
    
    R = np.array([
        [1 - 2*(y**2 + z**2),       2*(x*y - w*z),       2*(x*z + w*y)],
        [      2*(x*y + w*z), 1 - 2*(x**2 + z**2),       2*(y*z - w*x)],
        [      2*(x*z - w*y),       2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])
    return R

def rot_2_quat(R: np.ndarray):
    """
    Convert a rotation matrix to a quaternion. this route of coversion can be a bit ambiguous and numerically unstable. Algorithm from: [https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf]
    """
    if (R[2, 2] < 0):
        if (R[0, 0] > R[1, 1]):
            t = 1 + R[0, 0] - R[1, 1] - R[2, 2]
            quat = np.array([t, (R[0, 1] + R[1, 0]), (R[2, 0] + R[0, 2]), (R[1, 2] - R[2, 1])])
        else:
            t = 1 - R[0, 0] + R[1, 1] - R[2, 2]
            quat = np.array([(R[0, 1] + R[1, 0]), t, (R[1, 2] + R[2, 1]), (R[2, 0] - R[0, 2])])
    else:
        if (R[0, 0] < -R[1, 1]):
            t = 1 - R[0, 0] - R[1, 1] + R[2, 2]
            quat = np.array([(R[2, 0] + R[0, 2]), (R[1, 2] + R[2, 1]), t, (R[0, 1] - R[1, 0])])
        else:
            t = 1 + R[0, 0] - R[1, 1] + R[2, 2]
            quat = np.array([(R[1, 2] - R[2, 1]), (R[2, 0] - R[0, 2]), (R[0, 1] - R[1, 0]), t])
    
    quat = quat * (0.5/np.sqrt(t))
    return quat

