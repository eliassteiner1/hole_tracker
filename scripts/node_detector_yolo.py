#!/usr/bin/env python3
import os, sys, time, math
from   typing import Any
import cv2
import numpy as np

import rospy
import rospkg
from   cv_bridge          import CvBridge
from   sensor_msgs.msg    import CompressedImage
from   geometry_msgs.msg  import Point
from   hole_tracker.msg   import DetectionPoints # custom built

from   utils.multi_framework_yolo import DetectorMultiFramework
from   utils.utils                import generic_startup_log
from   utils.utils                import annotate_txt, annotate_crc


rospack = rospkg.RosPack()
Bridge  = CvBridge()

class NodeDetectorYolo:
    def __init__(self):
        """ main tracker node. Runs the HoleTracker, handles aggregating all the messages and fetches the depth estimate. Finally publishes the 1 tracker estimate point. """
        
        rospy.init_node("detector_yolo")
        self._get_params()
        
        self.Rate     = rospy.Rate(self.run_hz)
        self.Detector = DetectorMultiFramework(framework=self.framework, path=self.nnpath, minconf=self.minconf)
        
        # callback buffering
        self.buffer_image        = None
        self.buffer_image_newflg = False

        # setup subs & pubs last so that all other needed members are initialized for the callbacks
        self.PubDetections = rospy.Publisher("output_points", DetectionPoints, queue_size=1)
        self.PubImgdebug   = rospy.Publisher("output_img", CompressedImage, queue_size=1) # needs /compressed subtopic!
        self.SubImage      = rospy.Subscriber("input_img", CompressedImage, self._cllb_SubImage, queue_size=1)
        
        self._run()
    
    def _get_params(self):
        pkg_pth = rospack.get_path("hole_tracker")
        
        # load ros params from server
        prm_node = rospy.get_param("hole_tracker/node_detector_yolo")
        
        # extract params
        self.run_hz    = prm_node["run_hz"]
        self.framework = prm_node["framework"]
        self.nnpath    = os.path.join(pkg_pth, prm_node["nnpath"])
        self.minconf   = prm_node["minconf"]
        self.showdebug = prm_node["showdebug"]
        
        if self.framework not in ["ultralytics", "tensorrt"]:
            raise ValueError(f"choose a valid detector framework from [ultralytics, tensorrt]")
        
    def _cllb_SubImage(self, data):
        """this callback just stores the newest mesage in an intermediate buffer"""
        self.buffer_image        = data
        self.buffer_image_newflg = True
        
    def _process_image(self):
        """takes the newest buffered message, runs a yolo detection on it and publishes n detection points. The header of the original incoming image message is feedworwarded"""
        
        try: 
            # handle message input
            header = self.buffer_image.header 
            np_arr = np.frombuffer(self.buffer_image.data, np.uint8) # decode the compressed image to openCV format
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)          # decode the compressed image to openCV format

            # create the new message container for publishing n detection points
            detection_msg        = DetectionPoints()
            detection_msg.header = header
            detection_msg.points = []
            
            # yolo inference and point coordinate extraction
            points = self.Detector.detect(image) # returns [N, 2]
            
            for p in points:
                detection_msg.points.append(Point(x=p[0], y=p[1], z=0))

            if self.showdebug is True:
                for p in points:
                    annotate_crc(image, (p[0], p[1]), 15, (  0,   0,   0), 8)
                    annotate_crc(image, (p[0], p[1]), 15, (180, 100, 255), 5)
                
                # NOTE: in order for compressed image to be visible in rviz, publish under a /compressed subtopic!
                imgdebug_msg = Bridge.cv2_to_compressed_imgmsg(image, dst_format = "jpg")
                self.PubImgdebug.publish(imgdebug_msg)
            
            return detection_msg
            
        except Exception as e:
            rospy.logerr(f"Error during YOLO detection: {e}")
            return None
    
    def _startup_log(self):
        
        param_list = [
            dict(name= "run_hz",    value = self.run_hz,                   suffix = "Hz"),
            dict(name= "framework", value = self.framework,                suffix = None),
            dict(name= "nnpath",    value = os.path.basename(self.nnpath), suffix = None),
            dict(name= "minconf",   value = self.minconf,                  suffix = None),
            dict(name= "showdebug", value = self.showdebug,                suffix = None),
        ]
        
        rospy.loginfo(generic_startup_log("Yolo Detector", param_list, column_width = 80 ))
                      
    def _run(self):
        """automatically runs the node. Processes as many images as possible, limited by either compute ressources or run_hz frequency. Unprocessed image messages are discarded, only the most recent one is processed"""
        
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
        node = NodeDetectorYolo()
    except rospy.ROSInterruptException:
        pass
