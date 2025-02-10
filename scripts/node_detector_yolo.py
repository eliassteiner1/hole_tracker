#!/usr/bin/env python3
import os, sys, time, math
import cv2
import numpy as np

import rospy
import rospkg
from   cv_bridge          import CvBridge
from   sensor_msgs.msg    import CompressedImage
from   geometry_msgs.msg  import Point
from   hole_tracker.msg   import DetectionPoints # custom built!

from   utils.multi_framework_yolo import DetectorMultiFramework
from   utils.image_tools import ImageTools

bridge    = CvBridge()
rospack   = rospkg.RosPack()
Converter = ImageTools()

class NodeDetectorYolo():
    def __init__(self):
        
        rospy.init_node("detector_yolo")
        self._get_params()
        
        self.Rate          = rospy.Rate(self.run_hz)
        self.Detector      = DetectorMultiFramework(framework=self.framework, path=self.nnpath, minconf=self.minconf)
        
        self.PubDetections = rospy.Publisher("output_points", DetectionPoints, queue_size=1)
        self.PubImgdebug   = rospy.Publisher("output_img", CompressedImage, queue_size=1) # needs /compressed subtopic!
        self.SubImage      = rospy.Subscriber("input_img", CompressedImage, self._callback_SubImage, queue_size=1)
        
        # callback buffering
        self.buffer_image           = None
        self.buffer_image_newflg    = False

        self._run()
    
    def _get_params(self):
        pkg_pth = rospack.get_path("hole_tracker")
        
        self.run_hz    = rospy.get_param("~run_hz", 1.0)
        self.framework = rospy.get_param("~framework", "ultralytics")
        self.nnpath    = os.path.join(pkg_pth, rospy.get_param("~nnpath", "nnets/weights/augmented_holes_2.pt"))
        self.minconf   = rospy.get_param("~minconf", 0.0001)
        self.showdebug = rospy.get_param("~showdebug", True)
        
    def _callback_SubImage(self, data):
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
                    image = cv2.circle(
                        img       = image,
                        center    = (round(p[0]), round(p[1])),
                        radius    = 15,
                        color     = (255, 0, 255),
                        thickness = 5,
                    )
                
                # NOTE: in order for compressed image to be visible in rviz, publish under a /compressed subtopic!
                imgdebug_msg = Converter.convert_cv2_to_ros_compressed_msg(image, compressed_format="jpeg")
                self.PubImgdebug.publish(imgdebug_msg)
            
            return detection_msg
            
        except Exception as e:
            rospy.logerr(f"Error during YOLO detection: {e}")
            return None
                
    def _run(self):
        """automaticall runs the node. Processes as many images as possible, limited by either compute ressources or run_hz frequency. Unprocessed image messages are discarded, only the most recent one is processed"""
        
        rospy.loginfo(
            f"\n"
            f"------------------------------------------------------------------ \n"
            f"starting YOLO detector node with: \n\n"
            f"[runhz = {self.run_hz}] [minconf = {self.minconf}] [showdebug = {self.showdebug}] \n"
            f"------------------------------------------------------------------ \n"
            )
        
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
        # rospy.set_param("/use_sim_time", True) # use the simulated bag wall clock
        node = NodeDetectorYolo()
    except rospy.ROSInterruptException:
        pass