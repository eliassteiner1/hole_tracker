#!/usr/bin/env python3
import os, sys, time
import cv2
import copy
import numpy as np

import rospy          
from   cv_bridge         import CvBridge
from   geometry_msgs.msg import PointStamped, PoseStamped
from   nav_msgs.msg      import Odometry
from   sensor_msgs.msg   import CompressedImage
from   hole_tracker.msg  import DetectionPoints # custom built
from   omav_msgs.msg     import UAVStatus # custom built
from   hole_tracker.msg  import DepthMap # custom built

from   utils.equidistant_distorter import EquidistantDistorter
from   utils.hole_tracker          import HoleTracker, imu_interpolation
from   utils.utils                 import construct_camera_intrinsics, quat_2_rot, rot_2_quat
from   utils.transformations       import get_T_tof2imu, get_T_cam2imu
from   utils.utils                 import generic_startup_log
from   utils.utils                 import annotate_txt, annotate_crc


Bridge = CvBridge()

class NodeTracker:
    def __init__(self):
        
        rospy.init_node("tracker")
        self._get_params()
        
        self.Rate      = rospy.Rate(self.run_hz)
        self.Distorter = EquidistantDistorter(self.k1, self.k2, self.k3, self.k4)
        self.TRACKER   = HoleTracker(
            freq_visibility_check = self.freq_inframe_check,
            freq_memory_check     = self.freq_memory_check,
            freq_publish_estimate = self.freq_publish_estim,
            
            logging_level         = self.tracker_logging_lvl,
            tiebreak_method       = self.tracker_tiebreak_m,
            update_method         = self.tracker_update_m,
            
            thr_ddetect           = self.tracker_thr_ddetect,
            thr_imugaps           = self.tracker_thr_imugaps, 
            thr_inframe           = self.tracker_thr_inframe,
            thr_offrame           = self.tracker_thr_offrame,
            
            imu_hist_minlen       = self.tracker_imu_hist_minlen,
            imu_hist_maxlen       = self.tracker_imu_hist_maxlen,  
        )

        self.H         = construct_camera_intrinsics(self.f_hat_x, self.f_hat_y, self.cx, self.cy)
        self.H_INV     = np.linalg.inv(self.H)
        self.T_tof2imu = get_T_tof2imu()[0] # constant
        self.T_imu2tof = get_T_tof2imu()[1] # constant
        self.T_cam2imu = get_T_cam2imu()[0] # function of arm-angle
        self.T_imu2cam = get_T_cam2imu()[1] # function of arm-angle
        
        # message buffers that have to be handled directly / ASAP
        self.buffer_detections     = None
        self.buffer_detections_flg = False
        self.buffer_imu            = []
        self.buffer_imu_flg        = False
        
        # message buffers that just have to be available
        self.buffer_image          = None  
        self.buffer_image_flg      = False  
        self.buffer_image_dec      = None
        self.buffer_image_ts       = None
         
        self.buffer_uavstate          = None
        if self.depth_framework == "NORMAL": 
            self.buffer_normal        = None
        if self.depth_framework == "FUSION":
            self.buffer_depthmap      = None
            self.buffer_xu_lookup_map = None
            self.buffer_yv_lookup_map = None

        # flags for timer events
        self.do_inframe_check_flg    = False
        self.do_memory_check_flg     = False
        self.do_publish_imgdebug_flg = False
        
        # initialize subscribers last, otherwise messages will come in before other members are defined for callbacks
        self.SubDetections = rospy.Subscriber("input_points", DetectionPoints, self._cllb_sub_detections, queue_size=1)
        self.SubImage      = rospy.Subscriber("input_img", CompressedImage, self._cllb_sub_image, queue_size=1)
        self.SubImu        = rospy.Subscriber("input_odom", Odometry, self._cllb_sub_imu, queue_size=1)
        self.SubUavstate   = rospy.Subscriber("input_uavstate", UAVStatus, self._cllb_sub_uavstate, queue_size=1) 
        if self.depth_framework == "NORMAL":
            self.SubNormal = rospy.Subscriber("input_normal", PoseStamped, self._cllb_sub_normal,queue_size=1)
        if self.depth_framework == "FUSION":
            self.SubDepthMap = rospy.Subscriber("input_depthmap", DepthMap, self._cllb_sub_depthmap, queue_size=1)
        
        self.PubImgdebug = rospy.Publisher("output_img", CompressedImage, queue_size=1) # needs /compressed subtopic!
        self.PubEstimate = rospy.Publisher("output_estim", PointStamped, queue_size=1)
        
        self._run()
    
    def _get_params(self):
        # load ros params from server
        prm_node           = rospy.get_param("hole_tracker/node_tracker")
        prm_filter         = rospy.get_param("hole_tracker/filter_tracker")
        prm_rgb_resolution = rospy.get_param("hole_tracker/rgb_resolution")
        prm_rgb_intrinsics = rospy.get_param("hole_tracker/rgb_intrinsics")
        prm_rgb_distortion = rospy.get_param("hole_tracker/rgb_distortion")
        
        # extract params
        self.run_hz                  = prm_node["run_hz"]
        self.freq_inframe_check      = prm_node["freq_inframe_check"]
        self.freq_memory_check       = prm_node["freq_memory_check"]
        self.freq_publish_estim      = prm_node["freq_publish_estim"]
        self.freq_publish_imgdebug   = prm_node["freq_publish_imgdebug"]
        self.depth_framework         = prm_node["depth_framework"]
    
        self.tracker_logging_lvl     = prm_filter["logging_level"]
        self.tracker_tiebreak_m      = prm_filter["tiebreak_method"]
        self.tracker_update_m        = prm_filter["update_method"]
        self.tracker_thr_ddetect     = prm_filter["thr_ddetect"] # m
        self.tracker_thr_imugaps     = prm_filter["thr_imugaps"] # s
        self.tracker_thr_inframe     = prm_filter["thr_inframe"] # s
        self.tracker_thr_offrame     = prm_filter["thr_offrame"] # S
        self.tracker_imu_hist_minlen = prm_filter["imu_hist_minlen"] # nr of readings
        self.tracker_imu_hist_maxlen = prm_filter["imu_hist_maxlen"] # nr of readings
        
        self.cam_res                 = (prm_rgb_resolution[0], prm_rgb_resolution[1])
        
        self.f_hat_x                 = prm_rgb_intrinsics["f_hat_x"]
        self.f_hat_y                 = prm_rgb_intrinsics["f_hat_y"]
        self.cx                      = prm_rgb_intrinsics["cx"]
        self.cy                      = prm_rgb_intrinsics["cy"]
        
        self.k1                      = prm_rgb_distortion["k1"]
        self.k2                      = prm_rgb_distortion["k2"]
        self.k3                      = prm_rgb_distortion["k3"]
        self.k4                      = prm_rgb_distortion["k4"]
        
        if self.depth_framework not in ["NORMAL", "FUSION"]:
            raise ValueError(f"choose a valid depth framework from: [NORMAL, FUSION]")

    def _cllb_sub_detections(self, data): # just message buffer
        self.buffer_detections     = data
        self.buffer_detections_flg = True 

    def _cllb_sub_imu(self, data): # just message buffer
        ts  = data.header.stamp.to_nsec()/(10**9)
        lin = data.twist.twist.linear
        ang = data.twist.twist.angular
        
        self.buffer_imu.append([ts, lin.x, lin.y, lin.z, ang.x, ang.y, ang.z])        
        self.buffer_imu_flg = True

    def _cllb_sub_image(self, data): # just message buffer
        self.buffer_image     = data 
        self.buffer_image_flg = True
                
    def _cllb_sub_uavstate(self, data): # just message buffer
        """take: data.motors[6].position for cam arm angle position!"""    
        if np.isnan(data.motors[6].position):
            print(f"got a NaN value for servo position!")
        else:
            self.buffer_uavstate = data.motors[6].position
    
    def _cllb_sub_normal(self, data): # just message buffer
        pos  = data.pose.position
        quat = data.pose.orientation
        
        norm_p    = np.array([pos.x, pos.y, pos.z])
        norm_zvec = quat_2_rot(quat.x, quat.y, quat.z, quat.w)[:, 2]

        self.buffer_normal = (norm_p, norm_zvec) # normal estimation in TOF frame!
        
    def _cllb_sub_depthmap(self, data):
        self.buffer_xu_lookup_map = np.array(data.xu_lookup.data)
        self.buffer_yv_lookup_map = np.array(data.yv_lookup.data)
        self.buffer_depthmap      = Bridge.imgmsg_to_cv2(data.depthmap, desired_encoding="passthrough")
    
    def _timer_inframe_check(self, event): # these just set the flag
        self.do_inframe_check_flg = True
    
    def _timer_memory_check(self, event): # these just set the flag
        self.do_memory_check_flg = True
    
    def _timer_publish_imgdebug(self, event): # these just set the flag
        self.do_publish_imgdebug_flg = True    
     
    def _timer_publish_estim(self, event): # actually do the publishing 
        
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

    def _do_publish_imgdebug(self): # actually do the publishing 
        self.do_publish_imgdebug_flg = False
        
        # TODO ad info about depth estim, uavstate, and odometry to output image! (just ok or not)
        # TODO really clean this up. atomize stuff, avoid code reuse, encapsulate, etc. this is pretty opaque
        
        if self.buffer_image is None:
            return
        
        if self.buffer_image_flg is True: # create new image to work with
            self.buffer_image_flg = False
            # then decode the newly arrived image message
            np_arr                = np.frombuffer(self.buffer_image.data, np.uint8)
            self.buffer_image_dec = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # decode the compressed image to ndarray
            self.buffer_image_ts  = self.buffer_image.header.stamp.to_nsec()/(10**9)
            
            annotate_txt(self.buffer_image_dec, f"latest raw detector points -", (20, 30), (255,   0,   0))
            annotate_txt(self.buffer_image_dec, f"last confirmed detection -",   (20, 60), (  0, 255,   0))
            annotate_txt(self.buffer_image_dec, f"current estimate -",           (20, 90), (  0,   0, 255))
            
        frame = copy.deepcopy(self.buffer_image_dec)

        if self.buffer_detections is not None: # add latest raw yolo detects
            
            for p in self.buffer_detections.points:
                annotate_crc(frame, (p.x, p.y), 24, (255, 0, 0), 3)

        
        if len(self.TRACKER._p_detection["p"]) > 0: # add internal detect
            last_detect_P = self.TRACKER._p_detection[-1]["p"].squeeze()
            
            # transform last detection to camera frame
            last_detect_P   = np.append(last_detect_P, 1)
            last_detect_P   = self.T_imu2cam(self.buffer_uavstate) @ last_detect_P
            last_detect_P   = last_detect_P[0:3]
            last_detect_P_z = last_detect_P[2] # also store the z of detection
            
            # normalize, distort and project last stored detection points
            last_detect_P      = last_detect_P / last_detect_P[2]           # normalize
            last_detect_P[0:2] = self.Distorter.distort(last_detect_P[0:2]) # distort
            last_detect_P      = self.H @ last_detect_P                     # project
            
            annotate_crc(frame, (last_detect_P[0], last_detect_P[1]), 16, (0, 255, 0), 3)
            annotate_txt(frame, f"z: {last_detect_P_z:.2f}m", (last_detect_P[0]+30, last_detect_P[1]+30), (0, 255, 0))

        # this evaluation can cause the estimate to be evaluated before its creation time! (teeechnically you'd have to look for the historical estimate that was valid at the point of the image ts...)
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
            
            annotate_crc(frame, (P[0], P[1]), 10, (0, 0, 255), -1)
            annotate_txt(frame, f"z: {P_z:.2f}m", (P[0]+30, P[1]+30), (0, 0, 255))
            
        
        t_delay = (rospy.Time.now().to_nsec()/(10**9)) - self.buffer_image_ts
        annotate_txt(frame, f"image delay: {t_delay:.3f}s", (20, frame.shape[0]-90), (255, 255, 255))
        
        annotate_txt(frame, f"is tracking: {self.TRACKER._flag_tracking}", (20, frame.shape[0]-120), (255, 255, 255))
        
        t_vis   = sum(1-self.TRACKER._visibility_hist)*(1/self.freq_inframe_check)
        txt     = f"time since last detect (offrame): {t_vis:.2f}/{self.tracker_thr_offrame:.2f}s"
        annotate_txt(frame, txt, (20, frame.shape[0]-30), (255, 255, 255))
        
        t_vis   = sum(self.TRACKER._visibility_hist)*(1/self.freq_inframe_check)
        txt     = f"time since last detect (inframe): {t_vis:.2f}/{self.tracker_thr_inframe:.2f}s"
        annotate_txt(frame, txt, (20, frame.shape[0]-60), (255, 255, 255))
        
        # publish debugging output frame
        # NOTE: in order for compressed image to be visible in rviz, publish under a /compressed subtopic!
        imgdebug_msg = Bridge.cv2_to_compressed_imgmsg(frame, dst_format = "jpg")
        self.PubImgdebug.publish(imgdebug_msg)
    
    def _startup_log(self):

        param_list = [  
            dict(name = "run_hz",                  value = self.run_hz,                  suffix = "Hz"),  
            dict(name = "freq_inframe_check",      value = self.freq_inframe_check,      suffix = "Hz"),
            dict(name = "freq_memory_check",       value = self.freq_memory_check,       suffix = "Hz"),
            dict(name = "freq_publish_estim",      value = self.freq_publish_estim,      suffix = "Hz"),
            dict(name = "freq_publish_imgdebug",   value = self.freq_publish_imgdebug,   suffix = "Hz"),
            dict(name = "depth_framework",         value = self.depth_framework,         suffix = None),    
                
            dict(name = "tracker_logging_level",   value = self.tracker_logging_lvl,     suffix = None),
            dict(name = "tracker_tiebreak_m",      value = self.tracker_tiebreak_m,      suffix = None),
            dict(name = "tracker_update_m",        value = self.tracker_update_m,        suffix = None),
            dict(name = "tracker_thr_ddetect",     value = self.tracker_thr_ddetect,     suffix = "m"),
            dict(name = "tracker_thr_imugaps",     value = self.tracker_thr_imugaps,     suffix = "s"),
            dict(name = "tracker_thr_inframe",     value = self.tracker_thr_inframe,     suffix = "s"),
            dict(name = "tracker_thr_offrame",     value = self.tracker_thr_offrame,     suffix = "s"),
            dict(name = "tracker_imu_hist_minlen", value = self.tracker_imu_hist_minlen, suffix = None),
            dict(name = "tracker_imu_hist_maxlen", value = self.tracker_imu_hist_maxlen, suffix = None), 
            
            dict(name = "cam_res",                 value = self.cam_res,                 suffix = None),
            dict(name = "f_hat_x",                 value = self.f_hat_x,                 suffix = None),
            dict(name = "f_hat_y",                 value = self.f_hat_y,                 suffix = None),
            dict(name = "cx",                      value = self.cx,                      suffix = None),
            dict(name = "cy",                      value = self.cy,                      suffix = None),
                                            
            dict(name = "k1",                      value = self.k1,                      suffix = None),
            dict(name = "k2",                      value = self.k2,                      suffix = None),
            dict(name = "k3",                      value = self.k3,                      suffix = None),
            dict(name = "k4",                      value = self.k4,                      suffix = None),
        ]
        
        rospy.loginfo(generic_startup_log("Tracker", param_list, column_width = 80))
    
    
    def _mainloop_process_detections(self):
        ...
        
    def _mainloop_process_imu(self):
        ...
        
    def _mainloop_inframe_check(self):
        ...
        
    def _mainloop_memory_check(self):
        ...
        
    def _mainloop_publish_imgdebug(self):
        # this is kind of a duplicate to do_publishimg_debug
        ...
        
    def _run(self):
        self._startup_log()
        
        # timers for time-based events
        rospy.Timer(rospy.Duration(1/self.freq_inframe_check),    self._timer_inframe_check)
        rospy.Timer(rospy.Duration(1/self.freq_memory_check),     self._timer_memory_check)
        rospy.Timer(rospy.Duration(1/self.freq_publish_estim),    self._timer_publish_estim)
        rospy.Timer(rospy.Duration(1/self.freq_publish_imgdebug), self._timer_publish_imgdebug)
        
        while not rospy.is_shutdown(): # -------------------------------------------------------------------------------
            
            # always just the newest available -------------------------------------------------------------------------
            # image, normalestim, uavstate

            # do handling when something new has arrived ---------------------------------------------------------------
            
            if self.buffer_detections_flg is True: # --------------------------------------------- process Detections
                self.buffer_detections_flg = False

                n_points = len(self.buffer_detections.points)
                if   (n_points <= 0):
                    pass
                elif (self.buffer_uavstate is None):
                    pass
                elif (self.depth_framework == "NORMAL") and (self.buffer_normal   is None):
                    pass
                elif (self.depth_framework == "FUSION") and (self.buffer_depthmap is None):
                    pass
                else:
                    # ========== reformat the detections to a numpy array from the message
                    keypoints = np.zeros((n_points, 3))
                    ts        = self.buffer_detections.header.stamp.to_nsec()/(10**9)
                    points    = self.buffer_detections.points
                    for i, p in enumerate(points):
                        x      = p.x
                        y      = p.y
                        P      = self.H_INV @ np.array([x, y, 1]) # apply inverse intrinsics (from pixel to image plane)
                        P[0:2] = self.Distorter.undistort(P[0:2]) # undistort (from distorted to undistorted img plane)
                        
                        keypoints[i, :] = P

                    # ========== test: get the Z from the DEPTH MAP
                    if self.depth_framework == "FUSION":
                        Z_depthmap = []

                        for p in keypoints:
                            idx = np.clip(
                                np.searchsorted(self.buffer_xu_lookup_map, p[0]) - 1, 
                                a_min = 0, 
                                a_max = len(self.buffer_xu_lookup_map) - 2
                                )
                            idy = np.clip(
                                np.searchsorted(self.buffer_yv_lookup_map, p[1]) - 1, 
                                a_min = 0, 
                                a_max = len(self.buffer_yv_lookup_map) - 2
                                )
                            z_single = self.buffer_depthmap[idy, idx]
                            Z_depthmap.append(z_single)
                        Z_depthmap = np.array(Z_depthmap)[:, None]
                        
                    # ========== old: get the Z values from normal estimation (extrapolating plane)
                    if self.depth_framework == "NORMAL":
                        norm_p      = np.append(self.buffer_normal[0], 1)[:, None]
                        norm_zvec   = np.append(self.buffer_normal[1], 0)[:, None]
                        norm_p      = self.T_imu2cam(self.buffer_uavstate) @ self.T_tof2imu @ norm_p
                        norm_zvec   = self.T_imu2cam(self.buffer_uavstate) @ self.T_tof2imu @ norm_zvec
                        
                        numerator   = - norm_zvec[0:3].T @ norm_p[0:3]
                        denominator = norm_zvec[0:3].T @ keypoints.T
                        Z_normal    = - (numerator / denominator).T
                    
                    # ========== finally multiply the projected keypoints with the corresponding Z values
                    if self.depth_framework == "NORMAL":
                        keypoints = keypoints * Z_normal
                    if self.depth_framework == "FUSION":
                        keypoints   = keypoints * Z_depthmap    
                    
                    # ========== transform to imu frame
                    keypoints = keypoints.T
                    keypoints = np.concatenate([keypoints, np.ones((1, keypoints.shape[1]))])
                    keypoints = self.T_cam2imu(self.buffer_uavstate) @ keypoints
                    keypoints = (keypoints.T)[:, 0:3]
                    
                    # ========== finally give the new detection points (3D, in imu frame) to the Tracker
                    self.TRACKER.do_new_detection_logic(ts=ts, detections=keypoints)

            if self.buffer_imu_flg is True: # ----------------------------------------------------------- process IMU
                
                # quickly convert the buffered imu and then reset the buffer
                imu_array           = np.array(self.buffer_imu[-10:]) # cap it to the last 10 measurements
                self.buffer_imu     = []
                self.buffer_imu_flg = False # reset flag last for multithreading safety!
                
                if imu_array.shape[0] > 1:
                    ts      = imu_array[-1, 0]
                    lin_raw = imu_array[:, [1, 2, 3]]
                    ang_raw = imu_array[:, [4, 5, 6]]
            
                    lin     = imu_interpolation(lin_raw)
                    ang     = imu_interpolation(ang_raw)
                else: 
                    ts      = imu_array[-1, 0]
                    lin     = imu_array[-1, [1, 2, 3]]
                    ang     = imu_array[-1, [4, 5, 6]]

                self.TRACKER.do_new_imu_logic(ts=ts, new_imu=np.concatenate([lin, ang])[None, :])

            # do handling when timer is due ----------------------------------------------------------------------------
            # (but can't be done in timer callback itself because thread safety (write)...)
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
                self.TRACKER.do_inframe_check(ts=ts, estim2img=_tf_estim2img, img_res=self.cam_res) 
                
            if self.do_memory_check_flg is True: # ----------------------------------------------------- Memory Check
                self.do_memory_check_flg = False
                
                ts = rospy.Time.now().to_nsec()/(10**9)
                self.TRACKER.do_memory_check(ts=ts)
            
            if self.do_publish_imgdebug_flg is True:
                self._do_publish_imgdebug()
            
            self.Rate.sleep() # ----------------------------------------------------------------------------------------
    
    
if __name__ == "__main__":
    try:
        node = NodeTracker() # starts node!
    except rospy.ROSInterruptException:
        pass

