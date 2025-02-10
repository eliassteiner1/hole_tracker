#!/usr/bin/env python3
import os, sys, time
import argparse
import cv2
import numpy as np

import rospy
from   cv_bridge          import CvBridge
from   sensor_msgs.msg    import CompressedImage
from   geometry_msgs.msg  import Point
from   hole_tracker.msg   import DetectionPoints # custom built!

from   utils.image_tools  import ImageTools


converter = ImageTools()

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
        thickness = 5,
        shift     = None
        )
    
    img = cv2.circle(
        img       = img, 
        center    = (round(p[0]), round(p[1])), 
        radius    = round(rad), 
        color     = col, 
        thickness = 2,
        shift     = None
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

class NodeDetectorBlob():
    def __init__(self, runhz, showimg):
        # config
        self.run_hz  = runhz
        self.showimg = showimg
        
        # utilities
        rospy.init_node("blob_node", anonymous=True)
        self.bridge   = CvBridge()
        self.Detector = initialize_blob_detector()
        
        self.Rate          = rospy.Rate(self.run_hz)
        self.PubDetections = rospy.Publisher(
            "/tracker/detector/points", 
            DetectionPoints, 
            queue_size=1
        )
        
        self.PubImgdebug = rospy.Publisher(
            "/tracker/detector/img/compressed",
            CompressedImage,
            queue_size = 1   
        ) # needs /compressed subtopic!
        
        self.SubImage      = rospy.Subscriber(
            "/quail/wrist_cam/image_raw/compressed", 
            CompressedImage, 
            self.callback_SubImage, 
            queue_size=1
        )
        
        # callback buffering
        self.buffer_image           = None
        self.buffer_image_newflg    = False

        self.run()
    
    def callback_SubImage(self, data):
        """this callback just stores the newest mesage in an intermediate buffer"""
        self.buffer_image        = data
        self.buffer_image_newflg = True
        
    def process_image(self):
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

            # ==================================================================
            keyP = self.Detector.detect(image)
            for idx, P in enumerate(keyP):
                x = P.pt[0]
                y = P.pt[1]
                
                point = Point(x=x, y=y, z=0)
                detection_msg.points.append(point)
                
                if self.showimg is True:
                    sz = P.size
                    image = img_ann_marker(image, (x, y), sz, (255, 255, 0))
            
            # ==================================================================     
            
            if self.showimg is True:
                # NOTE: in order for compressed image to be visible in rviz, publish under a /compressed subtopic!
                imgdebug_msg = converter.convert_cv2_to_ros_compressed_msg(image, compressed_format="jpeg")
                self.PubImgdebug.publish(imgdebug_msg)
            
            return detection_msg
            
        except Exception as e:
            rospy.logerr(f"Error during blob detection: {e}")
            return None
                
    def run(self):
        """automaticall runs the node. Processes as many images as possible, limited by either compute ressources or run_hz frequency. Unprocessed image messages are discarded, only the most recent one is processed"""
        
        rospy.loginfo(
            f"\n"
            f"------------------------------------------------------------------ \n"
            f"starting blob detector node with: \n\n"
            f"[runhz = {self.run_hz}] [showdebug = {self.showimg}] \n"
            f"------------------------------------------------------------------ \n"
            )
        
        while not rospy.is_shutdown():
            if self.buffer_image_newflg is False:
                self.Rate.sleep()
                continue
            
            self.buffer_image_newflg = False
            msg = self.process_image()
            if msg is not None:
                self.PubDetections.publish(msg)
 
            self.Rate.sleep()


if __name__ == "__main__":
    try:
        # rospy.set_param("/use_sim_time", True) # use the simulated bag wall clock
        node = NodeDetectorBlob(runhz=10, showimg=True) # starts node!
    except rospy.ROSInterruptException:
        pass