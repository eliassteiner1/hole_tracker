#!/usr/bin/env python3
import os, sys, time, math
from typing import Any
import cv2
import numpy as np

import rospy
import rospkg
from   sensor_msgs.msg    import CompressedImage
from   geometry_msgs.msg  import Point
from   hole_tracker.msg   import DetectionPoints # custom built!

from   utils.multi_framework_yolo import DetectorMultiFramework
from   utils.image_tools import ImageTools

rospack   = rospkg.RosPack()
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
        shift     = None
        )
    
    img = cv2.circle(
        img       = img, 
        center    = (round(p[0]), round(p[1])), 
        radius    = round(rad), 
        color     = col, 
        thickness = 5,
        shift     = None
        )

    return img

class NodeDetectorYolo:
    def __init__(self):
        
        rospy.init_node("detector_yolo")
        self._get_params()
        
        self.Rate          = rospy.Rate(self.run_hz)
        self.Detector      = DetectorMultiFramework(framework=self.framework, path=self.nnpath, minconf=self.minconf)
        
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
        
        self.run_hz    = rospy.get_param("~run_hz", 1.0)
        self.framework = rospy.get_param("~framework", "ultralytics")
        self.nnpath    = os.path.join(pkg_pth, rospy.get_param("~nnpath", "nnets/weights/DS_6_real_drone_footage.pt"))
        self.minconf   = rospy.get_param("~minconf", 0.0001)
        self.showdebug = rospy.get_param("~showdebug", True)
        
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
                    image = img_ann_marker(image, p, 15, (180, 100, 255))
                
                # NOTE: in order for compressed image to be visible in rviz, publish under a /compressed subtopic!
                imgdebug_msg = Converter.convert_cv2_to_ros_compressed_msg(image, compressed_format="jpeg")
                self.PubImgdebug.publish(imgdebug_msg)
            
            return detection_msg
            
        except Exception as e:
            rospy.logerr(f"Error during YOLO detection: {e}")
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
            "\n\n" + f"╔{' STARTING YOLO DETECTOR NODE ':═^{max_width-2}}╗" + "\n" + "\n" + 

            _format_string(max_width, "runhz", self.run_hz, "Hz")                  + "\n" + 
            _format_string(max_width, "framework", self.framework)                 + "\n" + 
            _format_string(max_width, "nnpath", os.path.basename(self.nnpath))     + "\n" + 
            _format_string(max_width, "minconf", self.minconf)                     + "\n" + 
            _format_string(max_width, "showdebug", str(self.showdebug))            + "\n" + 
            
            "\n" + f"╚{'═'*(max_width-2)}╝"                                        + "\n"
        )
                      
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