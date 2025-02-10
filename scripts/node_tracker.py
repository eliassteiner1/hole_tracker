#!/usr/bin/env python3
import os, sys, time
import argparse
import cv2
import copy
import numpy as np
from   sympy import symbols, Matrix, sin, cos, lambdify

from   utils.EquidistantDistorter import EquidistantDistorter
from   utils.HoleTracker_V2       import HoleTracker, StructuredDeque
from   utils.utils                import construct_camera_intrinsics, quat_2_rot, rot_2_quat
from   utils.transformations      import get_T_tof2imu, get_T_cam2imu

import rospy
from   cv_bridge          import CvBridge
from   geometry_msgs.msg  import PointStamped, PoseStamped
from   nav_msgs.msg       import Odometry
from   sensor_msgs.msg    import Image, CompressedImage
from   hole_tracker.msg   import DetectionPoints # custom built!
from   omav_msgs.msg      import UAVStatus       # custom built!

# debug pcl ====================================================================
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
from   sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class TrackerNode():
    def __init__(self):
        
        rospy.init_node("tracker_node", anonymous=True)
        self._get_params()
        
        self.Rate   = rospy.Rate(self.run_hz)
        self.bridge = CvBridge()

        self.SubDetections = rospy.Subscriber("input_points", DetectionPoints, self.cllb_SubDetections, queue_size=1)
        self.SubImage      = rospy.Subscriber("input_img", CompressedImage, self.cllb_SubImage, queue_size=1)
        self.SubImu        = rospy.Subscriber("input_odom", Odometry, self.cllb_SubImu, queue_size=1)
        self.SubNormal     = rospy.Subscriber("input_normal", PoseStamped, self.cllb_SubNormal,queue_size=1)
        self.SubUavstate   = rospy.Subscriber("input_uavstate", UAVStatus, self.cllb_SubUavstate, queue_size=1) 
        
        self.PubImgdebug   = rospy.Publisher("output_img", Image, queue_size=1)
        self.PubEstimate   = rospy.Publisher("output_estim", PointStamped, queue_size=1)
        
        self.Distorter = EquidistantDistorter(k1=-0.11717, k2=0.005431, k3=0.003128, k4=-0.007101)
        self.TRACKER   = HoleTracker(
            f_visibility_check = self.freq_inframe_check,
            f_memory_res_check = self.freq_memory_check,
            f_publish_estimate = self.freq_publish_estim,
            HISTORY_LEN     = self.tracker_history_len,
            TIEBREAK_METHOD = self.tracker_tiebreak_m,
            THRESH_DETECT   = self.tracker_thr_detect,  # m
            THRESH_IMUGAPS  = self.tracker_thr_imugaps, # s
            THRESH_INFRAME  = self.tracker_thr_inframe, # s
            THRESH_OFFRAME  = self.tracker_thr_offrame, # s
            LOGGING_LEVEL   = "INFO"
        )
        self.H         = construct_camera_intrinsics(f_hat_x=1210.19, f_hat_y=1211.66, cx=717.19, cy=486.47)
        self.H_INV     = np.linalg.inv(self.H)
        self.T_tof2imu = get_T_tof2imu()[0]
        self.T_imu2tof = get_T_tof2imu()[1]
        self.T_cam2imu = get_T_cam2imu()[0]
        self.T_imu2cam = get_T_cam2imu()[1]
        
        # message buffers that have to be handled ASAP
        self.buffer_detections     = None
        self.buffer_detections_flg = False
        self.buffer_imu            = None
        self.buffer_imu_flg        = False
        
        # message buffers that just have to be available
        self.buffer_image          = None  
        self.buffer_image_flg      = False  
        self.buffer_image_dec      = None
        self.buffer_image_ts       = None
         
        self.buffer_normal         = None
        self.buffer_uavstate       = None

        # flags for timer events
        self.do_inframe_check_flg    = False
        self.do_memory_check_flg     = False
        self.do_publish_imgdebug_flg = False
        
        self.run()
    
    def _get_params(self):
        self.run_hz                = rospy.get_param("~run_hz", 50) # runs the main loop at a max of this frequency
        self.freq_inframe_check    = rospy.get_param("~freq_inframe_check", 5)
        self.freq_memory_check     = rospy.get_param("~freq_memory_check", 5)
        self.freq_publish_estim    = rospy.get_param("~freq_publish_estim", 30)
        self.freq_publish_imgdebug = rospy.get_param("~freq_publish_imgdebug", 10)
    
        self.tracker_thr_detect  = rospy.get_param("~tracker_thr_detect",  0.1)  # m
        self.tracker_thr_imugaps = rospy.get_param("~tracker_thr_imugaps", 0.5) # s
        self.tracker_thr_inframe = rospy.get_param("~tracker_thr_inframe", 2.0) # s
        self.tracker_thr_offrame = rospy.get_param("~tracker_thr_offrame", 5.0) # S
    
        self.tracker_history_len = rospy.get_param("~tracker_history_len", 20)
        self.tracker_tiebreak_m  = rospy.get_param("~tracker_tiebreak_m", "kde-0.1")
    
    # these callbacks just buffer the messages
    def cllb_SubDetections(self, data):
        self.buffer_detections     = data
        self.buffer_detections_flg = True 

    def cllb_SubImu(self, data):
        self.buffer_imu            = data
        self.buffer_imu_flg        = True

    def cllb_SubImage(self, data):
        self.buffer_image     = data 
        self.buffer_image_flg = True

    def cllb_SubNormal(self, data):
        pos  = data.pose.position
        quat = data.pose.orientation
        
        norm_p    = np.array([pos.x, pos.y, pos.z])
        norm_zvec = quat_2_rot(quat.x, quat.y, quat.z, quat.w)[:, 2]

        self.buffer_normal = (norm_p, norm_zvec) # normal estimation in TOF frame!
                
    def cllb_SubUavstate(self, data):
        """take: data.motors[6].position for cam arm angle position!"""    
        self.buffer_uavstate = data.motors[6].position
    
    # these just set the flag
    def timer_inframe_check(self, event):
        self.do_inframe_check_flg = True
    
    def timer_memory_check(self, event):
        self.do_memory_check_flg = True
    
    def timer_publish_imgdebug(self, event):
        self.do_publish_imgdebug_flg = True    
    
    # these actually do the publishing   
    def timer_publish_estim(self, event):
        
        P  = self.TRACKER.get_tracker_estimate(ts = rospy.Time.now().to_nsec()/(10**9))
        if P is not None:
            P = P.squeeze()
            
            estimate_msg                 = PointStamped()
            estimate_msg.header.stamp    = rospy.Time.now()
            estimate_msg.header.frame_id = "quail"
            
            estimate_msg.point.x = P[0]
            estimate_msg.point.y = P[1]
            estimate_msg.point.z = P[2]
            
            self.PubEstimate.publish(estimate_msg)

    def do_publish_imgdebug(self):
        self.do_publish_imgdebug_flg = False
        
        if self.buffer_image is None:
            return
        
        if self.buffer_image_flg is True: # create new image to work with
            self.buffer_image_flg = False
            # then decode the newly arrived image message
            np_arr                = np.frombuffer(self.buffer_image.data, np.uint8)
            self.buffer_image_dec = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # decode the compressed image to ndarray
            self.buffer_image_ts  = self.buffer_image.header.stamp.to_nsec()/(10**9)
            
            self.buffer_image_dec = cv2.putText(
                img  = self.buffer_image_dec,
                text = f"latest raw YOLO detects -",
                org  = (20, 30),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color = (255, 0, 0),
                thickness = 2,
            )
            
            self.buffer_image_dec = cv2.putText(
                img  = self.buffer_image_dec,
                text = f"last confirmed detection -",
                org  = (20, 60),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color = (0, 255, 0),
                thickness = 2,
            )

            self.buffer_image_dec = cv2.putText(
                img  = self.buffer_image_dec,
                text = f"current estimate -",
                org  = (20, 90),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color = (0, 0, 255),
                thickness = 2,
            )
            
        frame = copy.deepcopy(self.buffer_image_dec)

        if self.buffer_detections is not None: # add latest raw yolo detects
            for p in self.buffer_detections.points:
                frame = cv2.circle(
                    img       = frame, 
                    center    = (round(p.x), round(p.y)), 
                    radius    = 24, 
                    color     = (255, 0, 0), 
                    thickness = 3,
                    ) 
        
        if len(self.TRACKER._p_detection["p"]) > 0: # add internal detect
            last_detect_P = self.TRACKER._p_detection["p"].squeeze()
            
            # transform last detection to camera frame
            last_detect_P   = np.append(last_detect_P, 1)
            last_detect_P   = self.T_imu2cam(self.buffer_uavstate) @ last_detect_P
            last_detect_P   = last_detect_P[0:3]
            last_detect_P_z = last_detect_P[2] # also store the z of detection
            # normalize, distort and project the last detection from cam coordinates to image plane coordinates
            
            # normalize, distort and project last stored detection points
            last_detect_P      = last_detect_P / last_detect_P[2]           # normalize
            last_detect_P[0:2] = self.Distorter.distort(last_detect_P[0:2]) # distort
            last_detect_P      = self.H @ last_detect_P                     # project
            
            frame = cv2.circle(
                img       = frame, 
                center    = (round(last_detect_P[0]), round(last_detect_P[1])), 
                radius    = 16, 
                color     = (0, 255, 0), 
                thickness = 3,
                shift     = None
                ) 
            
            frame =cv2.putText(
                img  = frame,
                text = f"z: {last_detect_P_z:.2f}m",
                org  = (round(last_detect_P[0]) + 30, round(last_detect_P[1]) + 30),
                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color     = (0, 255, 0),
                thickness = 2
            )

        P  = self.TRACKER.get_tracker_estimate(ts=self.buffer_image_ts) # @ the time when the buffer frame was made  
        if P is not None: # add current estimate
            # transform P to camera frame
            P   = np.append(P.squeeze(), 1)
            P   = self.T_imu2cam(self.buffer_uavstate) @ P
            P   = P[0:3]
            P_z = P[2]
            
            # normalize, distort and project the estimate from cam coordinates to image plane coordinates
            P      = P.squeeze() / P.squeeze()[2]   # normalize
            P[0:2] = self.Distorter.distort(P[0:2]) # distort
            P      = self.H @ P                     # project 
            
            frame = cv2.circle(
                img       = frame, 
                center    = (round(P[0]), round(P[1])), 
                radius    = 8, 
                color     = (0, 0, 255), 
                thickness = -1,
                shift     = None
                ) 
            
            frame =cv2.putText(
                img  = frame,
                text = f"z: {P_z:.2f}m",
                org  = (round(P[0]) + 30, round(P[1]) - 30),
                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color     = (0, 0, 255),
                thickness = 2
            )

        frame =cv2.putText( # delay information
            img  = frame,
            text = f"image delay: {(rospy.Time.now().to_nsec()/(10**9)) - self.buffer_image_ts:.3f}s",
            org  = (20, frame.shape[0] - 30),
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.8,
            color     = (255, 255, 255),
            thickness = 2
        )
        
        frame =cv2.putText( # is tracking flag
            img  = frame,
            text = f"is tracking: {self.TRACKER._flag_tracking}",
            org  = (20, frame.shape[0] - 60),
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.8,
            color     = (255, 255, 255),
            thickness = 2
        )
        
        # publish debugging output frame
        imgdebug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.PubImgdebug.publish(imgdebug_msg)
    
    def run(self):
        
        # init debug ==============================================       
        self.PubPCL = rospy.Publisher(
            "/tracker/kde_pointcloud",
            PointCloud2,
            queue_size=1
        )
        
        # ======================================================================
        
        rospy.loginfo(
            f"\n"
            f"------------------------------------------------------------------ \n"
            f"starting object tracker node with: \n\n"
            f"[runhz = {self.run_hz}] [inframehz = {self.freq_inframe_check}] [memoryhz = {self.freq_memory_check}] "
            f"[estimhz = {self.freq_publish_estim}] [imgdebughz = {self.freq_publish_imgdebug}] \n"
            f"[trackerparams = thr det: {self.tracker_thr_detect}m, thr imugap: {self.tracker_thr_imugaps}s, "
            f"thr inframe: {self.tracker_thr_inframe}s, thr offrame: {self.tracker_thr_offrame}s] \n"
            f"------------------------------------------------------------------ \n"
            )
        
        rospy.Timer(rospy.Duration(1/self.freq_inframe_check),    self.timer_inframe_check)
        rospy.Timer(rospy.Duration(1/self.freq_memory_check),     self.timer_memory_check)
        rospy.Timer(rospy.Duration(1/self.freq_publish_estim),    self.timer_publish_estim)
        rospy.Timer(rospy.Duration(1/self.freq_publish_imgdebug), self.timer_publish_imgdebug)
        
        while not rospy.is_shutdown(): # -------------------------------------------------------------------------------
            
            # always just the newest available -------------------------------------------------------------------------
            # image, normalestim, uavstate

            # do handling when something new has arrived ---------------------------------------------------------------
            if self.buffer_detections_flg is True: # --------------------------------------------- process Detections
                self.buffer_detections_flg = False
                
                # reformat the detections to a numpy array from the message
                n_points = len(self.buffer_detections.points)
                if (n_points > 0) and (self.buffer_normal is not None) and (self.buffer_uavstate is not None):
                    keypoints = np.zeros((n_points, 3))

                    ts     = self.buffer_detections.header.stamp.to_nsec()/(10**9)
                    points = self.buffer_detections.points
                    for i, p in enumerate(points):
                        x      = p.x
                        y      = p.y
                        P      = self.H_INV @ np.array([x, y, 1]) # unproject
                        P[0:2] = self.Distorter.undistort(P[0:2]) # undistort
                        
                        keypoints[i, :] = P

                    # sort detections by how centered they are
                    dist      = np.sum(keypoints[:, 0:2]**2, axis=1)
                    keypoints = keypoints[dist.argsort()]
                    
                    # multiply keypoints with proper Z depths!
                    norm_p    = np.append(self.buffer_normal[0], 1)[:, None]
                    norm_zvec = np.append(self.buffer_normal[1], 0)[:, None]
                    norm_p    = self.T_imu2cam(self.buffer_uavstate) @ self.T_tof2imu @ norm_p
                    norm_zvec = self.T_imu2cam(self.buffer_uavstate) @ self.T_tof2imu @ norm_zvec
                    
                    numerator   = - norm_zvec[0:3].T @ norm_p[0:3]
                    denominator = norm_zvec[0:3].T @ keypoints.T
                    Z           = - (numerator / denominator).T
                    keypoints   = keypoints * Z
                    
                    # transform to imu frame
                    keypoints = keypoints.T
                    keypoints = np.concatenate([keypoints, np.ones((1, keypoints.shape[1]))])
                    keypoints = self.T_cam2imu(self.buffer_uavstate) @ keypoints
                    keypoints = (keypoints.T)[:, 0:3]
                    
                    self.TRACKER.do_new_detection_logic(ts=ts, detections=keypoints)
                    
                    # debug publish the pointcloud (detection_cloud) ===========
                    if self.TRACKER.detection_cloud is not None: 
                        arr = self.TRACKER.detection_cloud
                        
                        # Define PointCloud2 fields (XYZ)
                        fields = [
                            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                        ]
                        
                        # Convert NumPy array to a list of tuples
                        point_list = [tuple(p) for p in arr]
                        
                        # Create PointCloud2 message
                        point_cloud_msg = pc2.create_cloud_xyz32(
                            header=rospy.Header(frame_id="quail"), 
                            points=point_list
                            )
                        point_cloud_msg.header.stamp = rospy.Time.now()
                        
                        self.PubPCL.publish(point_cloud_msg)
                    
                    # ==========================================================
                    
                
                else:
                    pass
            
            if self.buffer_imu_flg is True: # ----------------------------------------------------------- process IMU
                self.buffer_imu_flg = False
                
                ts  = self.buffer_imu.header.stamp.to_nsec()/(10**9)
                lin = self.buffer_imu.twist.twist.linear
                ang = self.buffer_imu.twist.twist.angular
                
                self.TRACKER.do_new_imu_logic(ts=ts, new_imu=np.array([[lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]]))

            # do handling when timer is due ----------------------------------------------------------------------------
            # (but can't be done in timer event becuase thread safety write...)
            if self.do_inframe_check_flg is True: # --------------------------------------------------- Inframe Check
                self.do_inframe_check_flg = False
                
                def _tf_estim2img(P: np.ndarray):
                    # transform to camera coordinate frame (from IMU frame)
                    P = np.append(P.squeeze(), 1)
                    P = self.T_imu2cam(self.buffer_uavstate) @ P 
                    P = P[0:3]
                    Z = P[2]
                    
                    # transform to image plane
                    P      = P / P[2]                       # normalize
                    P[0:2] = self.Distorter.distort(P[0:2]) # distort
                    P      = self.H @ P                     # project
                    
                    return P[0:2], Z
                ts = rospy.Time.now().to_nsec()/(10**9)
                self.TRACKER.do_inframe_check(ts=ts, estim2img=_tf_estim2img, img_res=(1440, 1080))    
                
            if self.do_memory_check_flg is True: # ----------------------------------------------------- Memory Check
                self.do_memory_check_flg = False
                
                ts = rospy.Time.now().to_nsec()/(10**9)
                self.TRACKER.do_memory_check(ts=ts)
            
            if self.do_publish_imgdebug_flg is True:
                self.do_publish_imgdebug()
            
            self.Rate.sleep() # ----------------------------------------------------------------------------------------
    
    
if __name__ == "__main__":
    try:
        # rospy.set_param("/use_sim_time", True) # use the simulated bag wall clock
        node = TrackerNode() # starts node!
    except rospy.ROSInterruptException:
        pass

