#!/usr/bin/env python3
import os, sys, time
from   datetime import datetime
import copy
import csv
from   typing import Any
import cv2
import numpy as np

import rospy
import rospkg
import tf

from geometry_msgs.msg import PointStamped

rospack = rospkg.RosPack()

class NodePointTF:
    def __init__(self):
        rospy.init_node('point_tf')
        self._get_params()
        
        # setup csv writing (opening once for efficiency)
        if self.write_csv is True:
            pkg_pth  = rospack.get_path("hole_tracker")
            timename = datetime.now().strftime("%Y%m%d_%H%M%S") # format YYYYMMDD_HHMMSS
            csv_pth = os.path.join(pkg_pth, "log", f"estim_tf_{timename}.csv")
            
            self.csv_file   = open(csv_pth, mode="w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["timestamp", "x", "y", "z"]) # write the header
        
        # initialize subscribers last, otherwise messages will come in before other members are defined for callbacks
        self.Listener    = tf.TransformListener()
        self.SubEstimate = rospy.Subscriber("input_estim", PointStamped, self._cllb_SubEstimate, queue_size=1)
    
        self._run()    
    
    def _get_params(self):
        self.target_frame = rospy.get_param("~target_frame", "wall")
        self.verbose      = rospy.get_param("~verbose", False)
        self.write_csv    = rospy.get_param("~write_csv", False)
        
    def _cllb_SubEstimate(self, Data):
        source_point              = Data
        source_point.header.stamp = rospy.Time(0) # use the latest transform
        
        try: 
            target_point              = self.Listener.transformPoint(self.target_frame, source_point)
            p                         = target_point.point
            
            if self.verbose is True:
                print(f"<estimate_tf> TFed estimate to {self.target_frame}: [x={p.x:.3f}] [y={p.y:.3f}] [z={p.z:.3f}]")
            
            t0 = time.perf_counter()
            if self.write_csv is True:
                self.csv_writer.writerow([rospy.Time.now().to_sec(), p.x, p.y, p.z])
            t1 = time.perf_counter()
            print(f"writing could be at: {1/(t1-t0):.2f}HZ")
                
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Transform failed: {str(e)}")
            
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
            "\n\n" + f"╔{' STARTING ESTIMATE TF NODE ':═^{max_width-2}}╗" + "\n" + "\n" + 
            _format_string(max_width, "target_frame", self.target_frame)         + "\n" +
            "\n" + f"╚{'═'*(max_width-2)}╝"                                      + "\n"
        )
        
    def _run(self):
        self._startup_log()
        rospy.spin()
                 
    def __del__(self):
        # destructor of class, use to properly close the file on shutdown
        if hasattr(self, "csv_file") and self.csv_file:
            self.csv_file.close()
    
if __name__ == "__main__":
    try:
        node = NodePointTF() # starts node!
    except rospy.ROSInterruptException:
        pass

