#!/usr/bin/env python3
import numpy as np
from   typing import Any, List, Tuple
import cv2


def construct_camera_intrinsics(f_hat_x: float, f_hat_y: float, cx: float, cy: float):
    """ convenience function to construct the camera intrinsics matrix
    
    args
    ----
    - `f_hat_x`: effective focal length in x-direction [px]
    - `f_hat_y`: effective focal length in y-direction [px]
    - `cx`:      principal point x-coordinate [px] 
    - `cy`:      principal point y-coordinate [px]  
    """
    
    intrinsics = np.array([
        [f_hat_x,       0,     cx],
        [      0, f_hat_y,     cy],
        [      0,       0,      1],
    ], dtype=np.float32)
    
    return intrinsics

def quat_2_rot(x: float, y: float, z: float, w: float):
    """ Convert a quaternion to a rotation matrix, standard implementation """
    
    R = np.array([
        [1 - 2*(y**2 + z**2),       2*(x*y - w*z),       2*(x*z + w*y)],
        [      2*(x*y + w*z), 1 - 2*(x**2 + z**2),       2*(y*z - w*x)],
        [      2*(x*z - w*y),       2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])
    return R

def rot_2_quat(R: np.ndarray):
    """ Convert a rotation matrix to a quaternion. this route of coversion can be a bit ambiguous and numerically unstable. Algorithm from: [https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf] """
    
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

def log_line(width: int, name: str, value: Any, suffix: str=""):
    """ convenience for nicely formatting and padding info strings for the tracker repr method 
    
    args
    ----
    - `width`:  output string length (column width)
    - `name`:   parameter name
    - `value`:  parameter value
    - `suffix`: optional suffix string
    """
    
    suffix_str  = f" {suffix}" if suffix else "" # ensure suffix has a leading space if it's not empty
    base_str    = f"{name} {value}{suffix_str}" # base string without dots
    dots_needed = width - len(base_str) - 3  # calculate avail. space for dots (-2 brackets) (-1 added space)

    # construct the final formatted string with variable amounts of dots
    return f"[{name} {'┄' * dots_needed} {value}{suffix_str}]"

def generic_startup_log(node_name: str, param_list: List[dict], column_width: int):
    """ convenience function to generate a nicely formatted startup message for any node and a list of parameters 
    
    args
    ----
    - `node_name`:    name of the started node 
    - `param_list`:   list of dicts with information about parameters that should be printed (name, value, suffix)
    - `column_width`: string length (column width)
    """
    
    node_name = " STARTING NODE: " + node_name + " "
    
    output  = "\n\n" + f"╔{node_name:═^{column_width-2}}╗" + "\n" + "\n"
    output += "\n".join(log_line(column_width, el["name"], el["value"], el["suffix"]) for el in param_list)
    output += "\n\n" + f"╚{'═'*(column_width-2)}╝" + "\n"
    
    return output

def annotate_txt(img: np.ndarray, txt: str, loc: Tuple, col: Tuple[np.uint8]):
    """ convenience function for annotating cv2 text to images
    
    args
    ----
    - `img`: the image to annotate (modified, by reference)
    - `txt`: string to annotate
    - `loc`: upper left corner position of the textbox
    - `col`: line color
    """ 
    
    cv2.putText(
        img       = img,
        text      = txt,
        org       = (round(loc[0]), round(loc[1])),
        fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.8,
        color     = col,
        thickness = 2
    )

def annotate_crc(img: np.ndarray, ctr: Tuple, rad: float, col: Tuple[np.uint8], thi = int):
    """ convenience function for annotating cv2 circles to images
    
    args
    ----
    - `img`: the image to annotate (modified, by reference)
    - `ctr`: center of the circle (x, y)
    - `rad`: radius of the circle
    - `col`: line color
    - `thi`: line thickness (-1 for full circle)
    """
    
    cv2.circle(
        img       = img,
        center    = (round(ctr[0]), round(ctr[1])),
        radius    = round(rad),
        color     = col,
        thickness = thi
    )


