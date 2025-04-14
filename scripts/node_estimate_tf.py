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
            self.csv_writer.writerow(["timestamp", "x", "y", "z"]) # write the header
        
        # initialize subscribers last, otherwise messages will come in before the callback methods are defined
        self.Listener    = tf.TransformListener()
        self.SubEstimate = rospy.Subscriber("input_estim", PointStamped, self._cllb_SubEstimate, queue_size=1)
    
        self._run()    
    
    def _get_params(self):
        
        # load ros params from server
        prm_node = rospy.get_param("hole_tracker/node_estimate_tf")
        
        # extract params
        self.target_frame = prm_node["target_frame"]
        self.verbose      = prm_node["verbose"]
        self.write_csv    = prm_node["write_csv"]
        
    def _cllb_SubEstimate(self, Data):
        source_point              = Data # point from the estimator
        source_point.header.stamp = rospy.Time(0) # make sure to use the latest transform
        
        try: 
            target_point = self.Listener.transformPoint(self.target_frame, source_point)
            p            = target_point.point
            
            if self.verbose is True:
                print(f"<estimate_tf> TFed estimate to {self.target_frame}: [x={p.x:.3f}] [y={p.y:.3f}] [z={p.z:.3f}]")
            
            if self.write_csv is True:
                self.csv_writer.writerow([rospy.Time.now().to_sec(), p.x, p.y, p.z])
                
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Transform failed: {str(e)}")
            
    def _startup_log(self):
        
        param_list = [  
            dict(name = "target_frame", value = self.target_frame, suffix = None),
            dict(name = "verbose",      value = self.verbose     , suffix = None),
            dict(name = "write_csv",    value = self.write_csv   , suffix = None),
        ]
        
        rospy.loginfo(generic_startup_log("Estimate Transformer", param_list, column_width = 80))
        
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

