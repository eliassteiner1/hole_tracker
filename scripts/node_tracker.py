#!/usr/bin/env python3
import os, sys, time
from typing import Any
import cv2
import copy
import numpy as np
from   sympy import symbols, Matrix, sin, cos, lambdify

from   utils.equidistant_distorter import EquidistantDistorter
from   utils.hole_tracker          import HoleTracker
from   utils.image_tools           import ImageTools
from   utils.utils                 import construct_camera_intrinsics, quat_2_rot, rot_2_quat
from   utils.transformations       import get_T_tof2imu, get_T_cam2imu

import rospy
from   geometry_msgs.msg  import PointStamped, PoseStamped
from   nav_msgs.msg       import Odometry
from   sensor_msgs.msg    import CompressedImage
from   hole_tracker.msg   import DetectionPoints # custom built!
from   omav_msgs.msg      import UAVStatus       # custom built!


Converter = ImageTools()

def imu_interpolation(points: np.ndarray):
    """ takes the last few imu measurements (either linear or angular velocity) and finds its principle axis of change (equivalent to linear regression in 3D) the last measurement point is then projected onto the linear interpolation to find the 'latest smoothed velocity estimate'
    
    Aargs
    -----
    - `points`: np.ndarray of shape (n, 3) with either linear or angular velocity measurements to be smoothed  
    """
    
    # center the points to their center of mass for SVD to work correctly
    center          = np.mean(points, axis=0)
    points_centered = points - center
    
    # perform SVD (numpy uses BLAS / LAPACK so this is pretty efficient)
    _, _, Vt = np.linalg.svd(points_centered, full_matrices=False, compute_uv=True)
    
    # get the principal direction as normal vector
    direction_norm = Vt[0] / np.linalg.norm(Vt[0], ord=2)
    
    # project the last point onto the principal direction (and re-offset by center!)
    projection = center + np.dot(points_centered[-1], direction_norm) * direction_norm
    
    return projection

class NodeTracker:
    def __init__(self):
        
        rospy.init_node("tracker", anonymous=True)
        self._get_params()
        
        self.Rate   = rospy.Rate(self.run_hz)
        self.Distorter = EquidistantDistorter(k1=-0.11717, k2=0.005431, k3=0.003128, k4=-0.007101)
        self.TRACKER   = HoleTracker(
            freq_visibility_check = self.freq_inframe_check,
            freq_memory_check     = self.freq_memory_check,
            freq_publish_estimate = self.freq_publish_estim,
            
            logging_level   = self.tracker_logging_lvl,
            tiebreak_method = self.tracker_tiebreak_m,
            update_method   = self.tracker_update_m,
            
            thr_ddetect = self.tracker_thr_ddetect,
            thr_imugaps = self.tracker_thr_imugaps, 
            thr_inframe = self.tracker_thr_inframe,
            thr_offrame = self.tracker_thr_offrame,
            
            imu_hist_minlen = self.tracker_imu_hist_minlen,
            imu_hist_maxlen = self.tracker_imu_hist_maxlen,  
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
        self.buffer_imu            = []
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
        
        # initialize subscribers last, otherwise messages will come in before other members are defined for callbacks
        self.SubDetections = rospy.Subscriber("input_points", DetectionPoints, self._cllb_SubDetections, queue_size=1)
        self.SubImage      = rospy.Subscriber("input_img", CompressedImage, self._cllb_SubImage, queue_size=1)
        self.SubImu        = rospy.Subscriber("input_odom", Odometry, self._cllb_SubImu, queue_size=1)
        self.SubNormal     = rospy.Subscriber("input_normal", PoseStamped, self._cllb_SubNormal,queue_size=1)
        self.SubUavstate   = rospy.Subscriber("input_uavstate", UAVStatus, self._cllb_SubUavstate, queue_size=1) 
        
        self.PubImgdebug   = rospy.Publisher("output_img", CompressedImage, queue_size=1) # needs /compressed subtopic!
        self.PubEstimate   = rospy.Publisher("output_estim", PointStamped, queue_size=1)
        
        self._run()
    
    def _get_params(self):
        self.run_hz                = rospy.get_param("~run_hz", 50) # main loop is run at max this freq.
        self.freq_inframe_check    = rospy.get_param("~freq_inframe_check", 5)
        self.freq_memory_check     = rospy.get_param("~freq_memory_check", 5)
        self.freq_publish_estim    = rospy.get_param("~freq_publish_estim", 30)
        self.freq_publish_imgdebug = rospy.get_param("~freq_publish_imgdebug", 10)
    
        self.tracker_logging_lvl   = rospy.get_param("~tracker_logging_lvl", "DEBUG")
        self.tracker_tiebreak_m    = rospy.get_param("~tracker_tiebreak_m", "FIRST")
        self.tracker_update_m      = rospy.get_param("~tracker_update_m", "REPLACE")
        
        self.tracker_thr_ddetect   = rospy.get_param("~tracker_thr_ddetect", 0.1) # m
        self.tracker_thr_imugaps   = rospy.get_param("~tracker_thr_imugaps", 0.5) # s
        self.tracker_thr_inframe   = rospy.get_param("~tracker_thr_inframe", 2.0) # s
        self.tracker_thr_offrame   = rospy.get_param("~tracker_thr_offrame", 5.0) # S
    
        self.tracker_imu_hist_minlen = rospy.get_param("~tracker_imu_hist_minlen", 100)
        self.tracker_imu_hist_maxlen = rospy.get_param("~tracker_imu_hist_maxlen", 2000)
    
    def _cllb_SubDetections(self, data): # just message buffer
        self.buffer_detections     = data
        self.buffer_detections_flg = True 

    def _cllb_SubImu(self, data): # just message buffer
        ts  = data.header.stamp.to_nsec()/(10**9)
        lin = data.twist.twist.linear
        ang = data.twist.twist.angular
        
        self.buffer_imu.append([ts, lin.x, lin.y, lin.z, ang.x, ang.y, ang.z])        
        self.buffer_imu_flg = True

    def _cllb_SubImage(self, data): # just message buffer
        self.buffer_image     = data 
        self.buffer_image_flg = True

    def _cllb_SubNormal(self, data): # just message buffer
        pos  = data.pose.position
        quat = data.pose.orientation
        
        norm_p    = np.array([pos.x, pos.y, pos.z])
        norm_zvec = quat_2_rot(quat.x, quat.y, quat.z, quat.w)[:, 2]

        self.buffer_normal = (norm_p, norm_zvec) # normal estimation in TOF frame!
                
    def _cllb_SubUavstate(self, data): # just message buffer
        """take: data.motors[6].position for cam arm angle position!"""    
        self.buffer_uavstate = data.motors[6].position
    
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
        
        # TODO ad info about depth estim, uavstate, and odometry! (just ok or not)
        
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
                text = f"latest raw detector points -",
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
            last_detect_P = self.TRACKER._p_detection[-1]["p"].squeeze()
            # TODO: now also adjust this to the new longer p_detection store
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
            
            frame = cv2.circle(
                img       = frame, 
                center    = (round(P[0]), round(P[1])), 
                radius    = 10, 
                color     = (0, 0, 255), 
                thickness = -1,
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
            org  = (20, frame.shape[0] - 90),
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.8,
            color     = (255, 255, 255),
            thickness = 2
        )
        
        frame =cv2.putText( # is tracking flag
            img  = frame,
            text = f"is tracking: {self.TRACKER._flag_tracking}",
            org  = (20, frame.shape[0] - 120),
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.8,
            color     = (255, 255, 255),
            thickness = 2
        )
        
        frame =cv2.putText( # nodetect inframe
            img  = frame,
            text = f"time since last detect (offrame): {sum(1-self.TRACKER._visibility_hist)*(1/self.freq_inframe_check):.2f}/{self.tracker_thr_offrame:.2f}s",
            org  = (20, frame.shape[0] - 30),
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.8,
            color     = (255, 255, 255),
            thickness = 2
        )        
        
        frame =cv2.putText( # nodetect inframe
            img  = frame,
            text = f"time since last detect (inframe): {sum(self.TRACKER._visibility_hist)*(1/self.freq_inframe_check):.2f}/{self.tracker_thr_inframe:.2f}s",
            org  = (20, frame.shape[0] - 60),
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.8,
            color     = (255, 255, 255),
            thickness = 2
        )
        
        # publish debugging output frame
        # NOTE: in order for compressed image to be visible in rviz, publish under a /compressed subtopic!
        imgdebug_msg = Converter.convert_cv2_to_ros_compressed_msg(frame, compressed_format="jpeg")
        self.PubImgdebug.publish(imgdebug_msg)
    
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
            "\n\n" + f"╔{' STARTING OBEJCT TRACKER NODE ':═^{max_width-2}}╗"              + "\n" + "\n" + 
            _format_string(max_width, "run_hz", self.run_hz, "Hz")                               + "\n" + 
            
            _format_string(max_width, "freq_inframe_check", self.freq_inframe_check, "Hz")       + "\n" +
            _format_string(max_width, "freq_memory_check", self.freq_memory_check, "Hz")         + "\n" +
            _format_string(max_width, "freq_publish_estim", self.freq_publish_estim, "Hz")       + "\n" +
            _format_string(max_width, "freq_publish_imgdebug", self.freq_publish_imgdebug, "Hz") + "\n" +

            _format_string(max_width, "tracker_logging_level", self.tracker_logging_lvl)         + "\n" +
            _format_string(max_width, "tracker_tiebreak_m", self.tracker_tiebreak_m)             + "\n" +
            _format_string(max_width, "tracker_update_m", self.tracker_update_m)                 + "\n" +
            
            _format_string(max_width, "tracker_thr_ddetect", self.tracker_thr_ddetect, "m")      + "\n" +
            _format_string(max_width, "tracker_thr_imugaps", self.tracker_thr_imugaps, "s")      + "\n" +
            _format_string(max_width, "tracker_thr_inframe", self.tracker_thr_inframe, "s")      + "\n" + 
            _format_string(max_width, "tracker_thr_offrame", self.tracker_thr_offrame, "s")      + "\n" +
            
            _format_string(max_width, "tracker_imu_hist_minlen", self.tracker_imu_hist_minlen)   + "\n" +
            _format_string(max_width, "tracker_imu_hist_maxlen", self.tracker_imu_hist_maxlen)   + "\n" +
            
            "\n" + f"╚{'═'*(max_width-2)}╝"                                                      + "\n"
        )

    def _run(self):
        self._startup_log()
        
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
                
                else:
                    pass
            
            if self.buffer_imu_flg is True: # ----------------------------------------------------------- process IMU
                self.buffer_imu_flg = False
                
                # quickly convert the buffered imu and then reset the buffer
                imu_array       = np.array(self.buffer_imu[-10:]) # cap it to the last 10 measurements
                self.buffer_imu = []
                
                if imu_array.shape[0] > 1:
                    ts      = imu_array[-1, 0]
                    lin_raw = imu_array[:, [1, 2, 3]]
                    ang_raw = imu_array[:, [4, 5, 6]]
            
                    lin = imu_interpolation(lin_raw)
                    ang = imu_interpolation(ang_raw)
                else: 
                    ts  = imu_array[-1, 0]
                    lin = imu_array[-1, [1, 2, 3]]
                    ang = imu_array[-1, [4, 5, 6]]

                self.TRACKER.do_new_imu_logic(ts=ts, new_imu=np.concatenate([lin, ang])[None, :])

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
                self._do_publish_imgdebug()
            
            self.Rate.sleep() # ----------------------------------------------------------------------------------------
    
    
if __name__ == "__main__":
    try:
        node = NodeTracker() # starts node!
    except rospy.ROSInterruptException:
        pass

