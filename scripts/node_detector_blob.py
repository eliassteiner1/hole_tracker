#!/usr/bin/env python3
import os, sys, time
from typing import Any
import argparse
import cv2
import numpy as np

import rospy
from   cv_bridge          import CvBridge
from   sensor_msgs.msg    import CompressedImage
from   geometry_msgs.msg  import Point
from   hole_tracker.msg   import DetectionPoints # custom built!

from   utils.image_tools  import ImageTools


Converter = ImageTools()

def img_ann_marker(img: np.ndarray, p: np.ndarray, rad: float, col: tuple):
    """
    wrapper for the cv2 circle for convenience

    img: cv2 compatible image (usually 3 channel np.ndarray)
    p:   the circle centerpoint in [u, v] coordinates
    rad: the radius of the circle
    col: the RGB color of the circle

    """

    img = cv2.circle(
        img       = img, 
        center    = (round(p[0]), round(p[1])), 
        radius    = round(rad), 
        color     = (0, 0, 0), 
        thickness = 8,
        )
    
    img = cv2.circle(
        img       = img, 
        center    = (round(p[0]), round(p[1])), 
        radius    = round(rad), 
        color     = col, 
        thickness = 5,
        )

    return img

def initialize_blob_detector():
    # get the parameter object for blob detectors
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area: [blob is aroooound 300]
    params.filterByArea        = False
    params.minArea             = 20
    params.maxArea             = 500

    # Filter by Circularity: [circle = 1, square = 0.785, line = 0]
    params.filterByCircularity = True
    params.minCircularity      = 0.8
    params.maxCircularity      = 1.0

    # Filter by Convexity: (basically ratio of area of convex hull vs actual area)
    params.filterByConvexity   = True
    params.minConvexity        = 0.90
    params.maxConvexity        = 1.0

    # Filter by Inertia: (ratio between long and short axis) [circle = 1, line = 0]
    params.filterByInertia     = True
    params.minInertiaRatio     = 0.5
    params.maxInertiaRatio     = 1.0

    # Filter by Color: (seems to be broken)
    params.filterByColor = False
    params.blobColor     = 255  # White blobs

    # Thresholding: the blob detector creates multiple binary thresholded images. 
    # (for converting color image to grayscale, which is done in the detection algorithm)
    params.thresholdStep = 20
    params.minThreshold  = 30
    params.maxThreshold  = 150

    # Miscellaneous
    params.minRepeatability    = 2
    params.minDistBetweenBlobs = 100
    
    return cv2.SimpleBlobDetector_create(params)

class NodeDetectorBlob:
    def __init__(self):

        rospy.init_node("detector_blob")
        self._get_params()
        
        self.Rate     = rospy.Rate(self.run_hz)
        self.Detector = initialize_blob_detector()
        
        # callback buffering
        self.buffer_image        = None
        self.buffer_image_newflg = False
        
        # setup subs & pubs last so that all other needed members are initialized for the callbacks
        self.PubDetections = rospy.Publisher("output_points", DetectionPoints, queue_size=1)
        self.PubImgdebug   = rospy.Publisher("output_img", CompressedImage, queue_size=1) # needs /compressed subtopic!
        self.SubImage      = rospy.Subscriber("input_img", CompressedImage, self._cllb_SubImage, queue_size=1)
        
        self._run()
    
    def _get_params(self):
        self.run_hz    = rospy.get_param("~run_hz", 1.0)
        self.showdebug = rospy.get_param("~showdebug", True)
    
    def _cllb_SubImage(self, data):
        """this callback just stores the newest mesage in an intermediate buffer"""
        self.buffer_image        = data
        self.buffer_image_newflg = True
        
    def _process_image(self):
        """takes the newest buffered message, runs a yolo detection on it and publishes n detection points. The header of the original incoming image message is feedworwarded"""
        
        try: 
            # handle message input
            header = self.buffer_image.header # store the incoming header to feed forward
            
            np_arr = np.frombuffer(self.buffer_image.data, np.uint8) # decode the compressed image to openCV format
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)          # decode the compressed image to openCV format

            # create the new message container for publishing n detection points
            detection_msg        = DetectionPoints()
            detection_msg.header = header
            detection_msg.points = []

            keyP = self.Detector.detect(image)
            for idx, P in enumerate(keyP):
                x = P.pt[0]
                y = P.pt[1]
                
                point = Point(x=x, y=y, z=0)
                detection_msg.points.append(point)
                
                if self.showdebug is True:
                    sz = P.size
                    image = img_ann_marker(image, (x, y), (sz + 3), (255, 255, 0))     
            
            if self.showdebug is True:
                # NOTE: in order for compressed image to be visible in rviz, publish under a /compressed subtopic!
                imgdebug_msg = Converter.convert_cv2_to_ros_compressed_msg(image, compressed_format="jpeg")
                self.PubImgdebug.publish(imgdebug_msg)
            
            return detection_msg
            
        except Exception as e:
            rospy.logerr(f"Error during blob detection: {e}")
            return None
    
    def _startup_log(self):
        max_width = 60
        
        def _format_string(width: int, name: str, value: Any, suffix: str=""):
            """ convenience for nicely formatting and padding info strings for the tracker repr method """
            
            suffix_str  = f" {suffix}" if suffix else "" # ensure suffix has a leading space if it's not empty
            base_str    = f"{name} {value}{suffix_str}" # base string without dots
            dots_needed = width - len(base_str) - 3  # calculate avail. space for dots (-2 brackets) (-1 added space)

            # construct the final formatted string with variable amounts of dots
            return f"[{name} {'┄' * dots_needed} {value}{suffix_str}]"
        
        rospy.loginfo(
            "\n\n" + f"╔{' STARTING BLOB DETECTOR NODE ':═^{max_width-2}}╗" + "\n" + "\n" + 

            _format_string(max_width, "runhz", self.run_hz, "Hz")                  + "\n" +
            _format_string(max_width, "showdebug", str(self.showdebug))            + "\n" +
            
            "\n" + f"╚{'═'*(max_width-2)}╝"                                        + "\n"
        )
                
    def _run(self):
        """ automatically runs the node. Processes as many images as possible, limited by either compute ressources or run_hz frequency. Unprocessed image messages are discarded, only the most recent one is processed """

        self._startup_log()
        
        while not rospy.is_shutdown():
            if self.buffer_image_newflg is False:
                self.Rate.sleep()
                continue
            
            self.buffer_image_newflg = False
            msg = self._process_image()
            if msg is not None:
                self.PubDetections.publish(msg)
 
            self.Rate.sleep()


if __name__ == "__main__":
    try:
        node = NodeDetectorBlob() # starts node!
    except rospy.ROSInterruptException:
        pass