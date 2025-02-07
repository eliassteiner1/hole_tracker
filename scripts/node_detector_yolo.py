#!/usr/bin/env python3
import os, sys, time
import argparse
import cv2
import numpy as np
from   ultralytics import YOLO

import rospy
from   cv_bridge          import CvBridge
from   sensor_msgs.msg    import CompressedImage, Image
from   geometry_msgs.msg  import Point
from   hole_tracker.msg   import DetectionPoints # custom built!


class Detector:
    
    def __init__(self, framework: str, path: str, minconf: float):
        
        if framework not in ["ultralytics", "tensorrt"]:
            raise ValueError(f"please specify a valid framework from [ultralytics, tensorrt]! (got {framework=})")
        
        self.framework = framework
        self.minconf   = minconf
        self.Net       = None
        
        if self.framework == "ultralytics":
            # import ultralytics conditional
            from ultralytics import YOLO
            self.Net = YOLO(path)
            pass
        
        if self.framework == "tensorrt":
            # import pycuda and tensorrt conditional
            # TODO implement
            
            pass
        
    def detect(self, input: np.ndarray):
        # TODO sanitize input
        
        if self.framework == "ultralytics":
            
            pred   = self.Net(input, conf=self.minconf, verbose=False)
            points = []
            for el in pred: #TODO why is this necessary?
                boxes = el.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
                    x              = (x1 + x2) / 2
                    y              = (y1 + y2) / 2
                    R              = ((x2 - x1) + (y2 - y1)) / 2
                    points.append([x, y])
            return np.array(points)
        
        if self.framework == "tensorrt":
            
            # prooobably resize input image to 640x640, make sure it's dimensions are 1x3x640y640
            # maybe make it specifyable
            # run inference
            # filter output tensor by confidence (minconf)
            # rescale these boxes to image coordinates
            
            pass
        
        return


class NodeDetectorYolo():
    def __init__(self, runhz, minconf, showdebug=False):
        # config
        self.run_hz       = runhz
        self.minconf      = minconf
        self.showdebug    = showdebug
        self.yolo_weights = "/home/ubuntu/catkin_ws/src/hole_tracker/scripts/YOLO_weights/augmented_holes_2.pt"
        
        # utilities
        rospy.init_node("yolo_node", anonymous=True)
        self.bridge = CvBridge()
        self.Rate          = rospy.Rate(self.run_hz)
        self.Detector      = YOLO(self.yolo_weights)
        self.PubDetections = rospy.Publisher(
            "/tracker/detector/points", 
            DetectionPoints, 
            queue_size=1
            )
        
        self.PubImgdebug = rospy.Publisher(
            "/tracker/detector/img",
            Image,
            queue_size = 1
            
        )
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
            header = self.buffer_image.header 
            np_arr = np.frombuffer(self.buffer_image.data, np.uint8) # decode the compressed image to openCV format
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)          # decode the compressed image to openCV format

            # create the new message container for publishing n detection points
            detection_msg        = DetectionPoints()
            detection_msg.header = header
            detection_msg.points = []
            
            # yolo inference and point coordinate extraction
            results = self.Detector(image, conf=self.minconf, verbose=False)
            for res in results:
                boxes = res.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
                    x              = (x1 + x2) / 2
                    y              = (y1 + y2) / 2
                    R              = ((x2 - x1) + (y2 - y1)) / 2
                    
                    point = Point(x = x, y = y, z = 0)
                    detection_msg.points.append(point)
                    
                    if self.showdebug is True:
                        image = cv2.circle(
                            img       = image,
                            center    = (round(x), round(y)),
                            radius    = round(R),
                            color     = (255, 0, 255),
                            thickness = 5,
                        )
            if self.showdebug is True:
                imgdebug_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                self.PubImgdebug.publish(imgdebug_msg)
            
            return detection_msg
            
        except Exception as e:
            rospy.logerr(f"Error during YOLO detection: {e}")
            return None
                
    def run(self):
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
            msg = self.process_image()
            if msg is not None:
                self.PubDetections.publish(msg)
 
            self.Rate.sleep()

def handle_argparse():
    filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith("__")] # filters out ros internal args

    parser = argparse.ArgumentParser(description="ROS node for running a YOLO hole detector network")
    parser.add_argument(
        "-r", "--runhz", type = float, default = 5, 
        help = "detector node will be capped to run at a maximum of this frequency"
        )
    parser.add_argument(
        "-c", "--minconf", type = float, default = 0.001, 
        help = "minimum confidence threshold for the yolo detector"
        )
    parser.add_argument(
        "-d", "--showdebug", type = bool, default = True, 
        help = "if set to true, node will show it's detections in a cv2 imshow for debugging"
        )
    
    args = parser.parse_args(filtered_args)  # Pass only relevant args
    return args.runhz, args.minconf, args.showdebug

if __name__ == "__main__":
    try:
        r, c, d = handle_argparse()
        node = NodeDetectorYolo(r, c, showdebug = d) # starts node!
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        pass