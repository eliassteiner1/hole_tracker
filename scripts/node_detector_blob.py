#!/usr/bin/env python3
import os, sys, time
from typing import Any
import cv2
import numpy as np

import rospy
from   cv_bridge          import CvBridge
from   sensor_msgs.msg    import CompressedImage
from   geometry_msgs.msg  import Point
from   hole_tracker.msg   import DetectionPoints # custom built

from   utils.utils import generic_startup_log
from   utils.utils import annotate_txt, annotate_crc


Bridge = CvBridge()

def initialize_blob_detector(
    filter_by_area: bool,
    min_area: float,
    max_area: float,
    filter_by_circularity: bool,
    min_circularity: float,
    max_circularity: float,
    filter_by_convexity: bool,
    min_convexity: float,
    max_convexity: float,
    filter_by_inertia: bool,
    min_inertia: float,
    max_inertia: float,
    threshold_step: float,
    min_threshold: float,
    max_threshold: float,
    min_repeatability: int,
    min_dist_between_blobs: float,
):
    # get the parameter object for blob detectors
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area: [blob is aroooound 300]
    params.filterByArea        = filter_by_area
    params.minArea             = min_area
    params.maxArea             = max_area

    # filter by circularity: [circle = 1, square = 0.785, line = 0]
    params.filterByCircularity = filter_by_circularity
    params.minCircularity      = min_circularity
    params.maxCircularity      = max_circularity

    # filter by convexity: (basically ratio of area of convex hull vs actual area)
    params.filterByConvexity   = filter_by_convexity
    params.minConvexity        = min_convexity
    params.maxConvexity        = max_convexity

    # filter by inertia: (ratio between long and short axis) [circle = 1, line = 0]
    params.filterByInertia     = filter_by_inertia
    params.minInertiaRatio     = min_inertia
    params.maxInertiaRatio     = max_inertia

    # filter by color: (seems to be broken)
    params.filterByColor       = False
    params.blobColor           = 255  # White blobs

    # thresholding: the blob detector creates multiple binary thresholded images. (for converting color image to grayscale, which is done in the detection algorithm)
    params.thresholdStep       = threshold_step
    params.minThreshold        = min_threshold
    params.maxThreshold        = max_threshold

    # miscellaneous: repeatability = how many times does the blob have to be detected among different theshold images, dist = minimal distance between two blobs to count as different detections
    params.minRepeatability    = min_repeatability
    params.minDistBetweenBlobs = min_dist_between_blobs
    
    return cv2.SimpleBlobDetector_create(params)

class NodeDetectorBlob:
    def __init__(self):

        rospy.init_node("detector_blob")
        self._get_params()
        
        self.Rate     = rospy.Rate(self.run_hz)
        self.Detector = initialize_blob_detector(
            self.filter_by_area,
            self.min_area,
            self.max_area,
            self.filter_by_circularity,
            self.min_circularity,
            self.max_circularity,
            self.filter_by_convexity,
            self.min_convexity,
            self.max_convexity,
            self.filter_by_inertia,
            self.min_inertia,
            self.max_inertia,
            self.threshold_step,
            self.min_threshold,
            self.max_threshold,
            self.min_repeatability,
            self.min_dist_between_blobs,
        )
        
        # callback buffering
        self.buffer_image        = None
        self.buffer_image_newflg = False
        
        # setup subs & pubs last so that all other needed members are initialized for the callbacks
        self.PubDetections = rospy.Publisher("output_points", DetectionPoints, queue_size=1)
        self.PubImgdebug   = rospy.Publisher("output_img", CompressedImage, queue_size=1) # needs /compressed subtopic!
        self.SubImage      = rospy.Subscriber("input_img", CompressedImage, self._cllb_SubImage, queue_size=1)
        
        self._run()
    
    def _get_params(self):
        
        # load ros params from server
        prm_node        = rospy.get_param("hole_tracker/node_detector_blob")
        prm_simple_blob = rospy.get_param("hole_tracker/simple_blob")
        
        # extract params
        self.run_hz                 = prm_node["run_hz"]
        self.showdebug              = prm_node["showdebug"]
        
        self.filter_by_area         = prm_simple_blob["filter_by_area"] 
        self.min_area               = prm_simple_blob["min_area"]
        self.max_area               = prm_simple_blob["max_area"]
        self.filter_by_circularity  = prm_simple_blob["filter_by_circularity"] 
        self.min_circularity        = prm_simple_blob["min_circularity"] 
        self.max_circularity        = prm_simple_blob["max_circularity"] 
        self.filter_by_convexity    = prm_simple_blob["filter_by_convexity"] 
        self.min_convexity          = prm_simple_blob["min_convexity"] 
        self.max_convexity          = prm_simple_blob["max_convexity"] 
        self.filter_by_inertia      = prm_simple_blob["filter_by_inertia"] 
        self.min_inertia            = prm_simple_blob["min_inertia"] 
        self.max_inertia            = prm_simple_blob["max_inertia"] 
        self.threshold_step         = prm_simple_blob["threshold_step"]
        self.min_threshold          = prm_simple_blob["min_threshold"]
        self.max_threshold          = prm_simple_blob["max_threshold"]
        self.min_repeatability      = prm_simple_blob["min_repeatability"]
        self.min_dist_between_blobs = prm_simple_blob["min_dist_between_blobs"]
    
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
                    annotate_crc(image, (x, y), (sz + 5), (  0,   0,   0), 9)
                    annotate_crc(image, (x, y), (sz + 5), (255, 255,   0), 6)

            if self.showdebug is True:
                # NOTE: in order for compressed image to be visible in rviz, publish under a /compressed subtopic!
                imgdebug_msg = Bridge.cv2_to_compressed_imgmsg(image, dst_format = "jpg")
                self.PubImgdebug.publish(imgdebug_msg)
            
            return detection_msg
            
        except Exception as e:
            rospy.logerr(f"Error during blob detection: {e}")
            return None
    
    def _startup_log(self):
        param_list = [
            dict(name = "run_hz",    value = self.run_hz,    suffix = "Hz"),
            dict(name = "showdebug", value = self.showdebug, suffix = None),
            
            dict(name = "filter_by_area ",         value = self.filter_by_area,         suffix = None),
            dict(name = "min_area ",               value = self.min_area,               suffix = None),
            dict(name = "max_area ",               value = self.max_area,               suffix = None),
            dict(name = "filter_by_circularity ",  value = self.filter_by_circularity,  suffix = None),
            dict(name = "min_circularity ",        value = self.min_circularity,        suffix = None),
            dict(name = "max_circularity ",        value = self.max_circularity,        suffix = None),
            dict(name = "filter_by_convexity ",    value = self.filter_by_convexity,    suffix = None),
            dict(name = "min_convexity ",          value = self.min_convexity,          suffix = None),
            dict(name = "max_convexity ",          value = self.max_convexity,          suffix = None),
            dict(name = "filter_by_inertia ",      value = self.filter_by_inertia,      suffix = None),
            dict(name = "min_inertia ",            value = self.min_inertia,            suffix = None),
            dict(name = "max_inertia ",            value = self.max_inertia,            suffix = None),
            dict(name = "threshold_step ",         value = self.threshold_step ,        suffix = None),
            dict(name = "min_threshold ",          value = self.min_threshold,          suffix = None),
            dict(name = "max_threshold ",          value = self.max_threshold,          suffix = None),
            dict(name = "min_repeatability ",      value = self.min_repeatability,      suffix = None),
            dict(name = "min_dist_between_blobs ", value = self.min_dist_between_blobs, suffix = None),   
        ]
        
        rospy.loginfo(generic_startup_log("Blob Detector", param_list, column_width = 80))
                
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