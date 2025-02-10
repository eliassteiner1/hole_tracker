#!/usr/bin/env python3
import numpy as np
import cv2


class EquidistantDistorter:
    def __init__(self, k1: float, k2: float, k3: float, k4: float):
        """
        set the camera calibration parameters k1-4 from the equidistant distortion model for wide-angle lenses
        """
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        return

    def distort(self, point: tuple):
        """
        [P normalized image plane] ==> [P distorted image plane]
        Apply equidistant distortion to a 2D point in the normalized image plane.
        
        Args:
            point: (X/Z, Y/Z) from [X, Y, Z]/Z (the 1 is left out, but only input normalized coords!)
        Returns:
            point: (X/Z', Y/Z') distorted point in normalized image plane
        """
        
        x, y = point
        r = np.sqrt(x**2 + y**2)
        
        if r < 1e-8:
            return np.array([x, y])  # No distortion for small radii

        theta = np.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        # Apply polynomial distortion
        thetad  = theta * (1 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8)
        scaling = thetad / r

        return np.array([x * scaling, y * scaling])

    def undistort(self, point: tuple, iterations: int=20):
        """
        [P distorted image plane] ==> [P normalized image plane]
        Remove equidistant distortion from a 2D point in the distorted image plane.

        Args:
            point: (X/Z', Y/Z') in distorted normalized image plane coords 
        Returns:
            point (X/Z, Y/Z) from [X, Y, Z]/Z (the 1 is left out, but only input normalized coords!)
        """
        
        x, y = point
        thetad = np.sqrt(x**2 + y**2)

        if thetad < 1e-8:
            return np.array([x, y])  # No need to undistort near the center

        theta = thetad  # Initial guess

        # Iterative undistortion
        for _ in range(iterations):
            theta2 = theta  * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta4 * theta4
            theta  = thetad / (1 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8)

        scaling = np.tan(theta) / thetad
        return np.array([x * scaling, y * scaling])