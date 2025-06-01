#!/usr/bin/env python3
import os, sys, time
import numpy as np
from   datetime import datetime
import csv
from   typing import Any

import rospy
import rospkg
import tf
from   geometry_msgs.msg import PointStamped

from   utils.utils import generic_startup_log


rospack = rospkg.RosPack()

class NodePointTF:
    def __init__(self):
        """ helper node for automatically transforming the tracker estimate into some other frame from the TF tree and store the transformed positions in a csv log. """
        
        rospy.init_node("point_tf")
        self._get_params()
        
        # setup csv writing (opening only once for efficiency)
        if self.write_csv is True:
            pkg_pth  = rospack.get_path("hole_tracker") # path to current ros package
            timename = datetime.now().strftime("%Y%m%d_%H%M%S") # format YYYYMMDD_HHMMSS
            csv_pth  = os.path.join(pkg_pth, "log", f"estim_tf_{timename}.csv")
             
            self.csv_file   = open(csv_pth, mode="w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["timestamp", "estim_x", "estim_y", "estim_z", "drone_x", "drone_y", "drone_z"])
        
        # initialize subscribers last, otherwise messages will come in before the callback methods are defined
        self.Listener    = tf.TransformListener()
        self.SubEstimate = rospy.Subscriber("input_estim", PointStamped, self._cllb_SubEstimate, queue_size=1)
    
        self._RUN()    
    
    def _get_params(self):
        
        # load ros params from server
        prm_node = rospy.get_param("hole_tracker/node_estimate_tf")
        
        # extract params
        self.target_frame = prm_node["target_frame"]
        self.verbose      = prm_node["verbose"]
        self.write_csv    = prm_node["write_csv"]
        
    def _cllb_SubEstimate(self, Data):
        # point from the estimator (source frame "quail")
        estimate_point              = Data 
        estimate_point.header.stamp = rospy.Time(0) # makes sure to use the latest transform
        
        # dummy point at drone origin to get it's location relative to the wall
        drone_point                 = PointStamped()
        drone_point.header.stamp    = rospy.Time(0)
        drone_point.header.frame_id = "quail" # source frame
        drone_point.point.x         = 0
        drone_point.point.y         = 0
        drone_point.point.z         = 0
        
        try: 
            # transform estimate point
            estimate_point_TFed = self.Listener.transformPoint(self.target_frame, estimate_point)
            pE                  = estimate_point_TFed.point
            
            # transform dummy drone origin point
            drone_point_TFed = self.Listener.transformPoint(self.target_frame, drone_point)
            pD               = drone_point_TFed.point
            
            if self.verbose is True:
                print(f"TFed estimate to {self.target_frame}: [x={pE.x:.3f}] [y={pE.y:.3f}] [z={pE.z:.3f}]")
                print(f"TFed drone location to {self.target_frame}: [x={pD.x:.3f}] [y={pD.y:.3f}] [z={pD.z:.3f}]")
            
            if self.write_csv is True:
                self.csv_writer.writerow([rospy.Time.now().to_sec(), pE.x, pE.y, pE.z, pD.x, pD.y, pD.z])
                
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Transform failed: {str(e)}")
            
    def _startup_log(self):
        
        param_list = [  
            dict(name = "target_frame", value = self.target_frame, suffix = None),
            dict(name = "verbose",      value = self.verbose     , suffix = None),
            dict(name = "write_csv",    value = self.write_csv   , suffix = None),
        ]
        
        rospy.loginfo(generic_startup_log("Estimate Transformer", param_list, column_width = 80))
        
    def _RUN(self):
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

