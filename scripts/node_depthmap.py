#!/usr/bin/env python3
import os, sys, time
import warnings
import copy
import cv2
import numpy as np

import rospy
from   cv_bridge         import CvBridge
from   sensor_msgs.msg   import PointCloud2
from   sensor_msgs.msg   import CompressedImage
from   omav_msgs.msg     import UAVStatus # custom built
from   std_msgs.msg      import Float32MultiArray
from   hole_tracker.msg  import DepthMap # custom built

from   utils.transformations import get_T_tof2imu, get_T_cam2imu
from   utils.utils           import construct_camera_intrinsics
from   utils.utils           import generic_startup_log

Bridge = CvBridge()

ROS_TO_NUMPY_DTYPE = {
    # Mapping of ROS PointField data types to NumPy dtype codes
    
    1:  'i1',  # int8     → 'i1'  (1-byte signed integer)
    2:  'u1',  # uint8    → 'u1'  (1-byte unsigned integer)
    3:  'i2',  # int16    → 'i2'  (2-byte signed integer)
    4:  'u2',  # uint16   → 'u2'  (2-byte unsigned integer)
    5:  'i4',  # int32    → 'i4'  (4-byte signed integer)
    6:  'u4',  # uint32   → 'u4'  (4-byte unsigned integer)
    7:  'f4',  # float32  → 'f4'  (4-byte floating point)
    8:  'f8'   # float64  → 'f8'  (8-byte floating point)
}

class NodeDepthMap:
    
    def __init__(self):
        rospy.init_node("depthmap_node")
        self._get_params()
        
        self.Rate       = rospy.Rate(self.run_hz)
        
        self.TF_TOF2IMU = get_T_tof2imu()[0] # to coordinate-transform from ToF to IMU frame
        self.TF_IMU2CAM = get_T_cam2imu()[1] # to coordinate transform from IMU to RGBcam frame (f of arm-angle)!
        
        self.CAM_H      = construct_camera_intrinsics(self.f_hat_x, self.f_hat_y, self.cx, self.cy)
        self.CAM_H_INV  = np.linalg.inv(self.CAM_H)
        self.CAM_RES    = self.CAM_RES # (w, h), determined from ros param server
         
        self.DS_FACTOR  = self.DS_FACTOR # from ros param server, only for the initial downsampling of the pointcloud
        self.Z_CUTOFF   = self.Z_CUTOFF  # from ros param server, this applies AFTER the homogenous transformation

        pmin = (self.CAM_H_INV @ np.array([0, 0, 1])[:, None]).squeeze()
        pmax = (self.CAM_H_INV @ np.array([self.CAM_RES[0]-1, self.CAM_RES[1]-1, 1])[:, None]).squeeze()
        self.BIN_PRM = dict(
            # the x and y ranges represent the coordinate ranges that are covered by the output depthmap in normalized image space. therefore, the limits are determined from the rgb cam intrinsics. this doesn't take distortion into account, so that just means that sooome corner points of the real image, when undistorted, could technically land outside the depthmap, but their values will just be inferred from the closest one ...
            x_range = [pmin[0], pmax[0]],
            y_range = [pmin[1], pmax[1]],
            
            # this is the range of depth values that is covered by the depthmap. the lower bound cannot be < 0 because for such values the camera projection that is used to generate the new depth image is undefined! in order to gain better precision for points close to the camera, while still using as few bins as possible (is costly) the range is square-rooted to have a more favourable distribution.
            z_range = self.z_range,
            
            # the number of bins for each dimension. x and y are determined from a desired resolution
            x_bins  = round((pmax[0] - pmin[0]) * self.binning_res),
            y_bins  = round((pmax[1] - pmin[1]) * self.binning_res),
            z_bins  = self.z_bins,
            
            # the "container width" for the bins in each dimension. used for np.histogramdd
            dx      = None,
            dy      = None,
            dz      = None,
        )
        self.BIN_PRM["z_range"][0] = (self.BIN_PRM["z_range"][0]**0.5) #  x^0.5 scaled for better precision distribution
        self.BIN_PRM["z_range"][1] = (self.BIN_PRM["z_range"][1]**0.5)
        self.BIN_PRM["dx"]         = (self.BIN_PRM["x_range"][1] - self.BIN_PRM["x_range"][0]) / self.BIN_PRM["x_bins"]
        self.BIN_PRM["dy"]         = (self.BIN_PRM["y_range"][1] - self.BIN_PRM["y_range"][0]) / self.BIN_PRM["y_bins"]
        self.BIN_PRM["dz"]         = (self.BIN_PRM["z_range"][1] - self.BIN_PRM["z_range"][0]) / self.BIN_PRM["z_bins"]
          
        self.do_init_flg     = True # to track whether the initialization has been done (once)
        self.TOF_H           = None # determined automatically (in initialization)
        self.TOF_H_INV       = None # determined automatically (in initialization)
        self.TOF_RES         = None # determined automatically (in initialization) (W, H)
        self.TOF_RES_SCALED  = None # determined automatically (in initialization) (W, H)
        self.TOF_XU_LOOKUP   = None # determined from TOFs Intrinsics
        self.TOF_YV_LOOKUP   = None # determined from TOFs Intrinsics

        # for input buffering
        self.buffer_pcl      = None # (H*W) x [X, Y, Z]
        self.buffer_pcl_flag = False
        self.buffer_armangle = None

        # setup publishers and subscribers last, so that all needed callback methods are defined already
        self.SubPcl      = rospy.Subscriber("input_pcl", PointCloud2, self._cllb_pointcloud, queue_size=1)
        self.SubUavstate = rospy.Subscriber("input_uavstate", UAVStatus, self._cllb_uavstate, queue_size=1) 
        self.PubImg      = rospy.Publisher("output_img", CompressedImage, queue_size=1)
        self.PubDepthMap = rospy.Publisher("output_depthmap", DepthMap, queue_size=1)

        self._RUN()
        
    def _get_params(self):
        
        # load ros params from server
        prm_node           = rospy.get_param("hole_tracker/node_depthmap")
        prm_rgb_resolution = rospy.get_param("hole_tracker/rgb_resolution")
        prm_rgb_intrinsics = rospy.get_param("hole_tracker/rgb_intrinsics")
        prm_pcl_processing = rospy.get_param("hole_tracker/pcl_processing")
            
        # extract params
        self.run_hz = prm_node["run_hz"]
        
        self.f_hat_x = prm_rgb_intrinsics["f_hat_x"]
        self.f_hat_y = prm_rgb_intrinsics["f_hat_y"]
        self.cx      = prm_rgb_intrinsics["cx"]
        self.cy      = prm_rgb_intrinsics["cy"]
        
        self.CAM_RES = (prm_rgb_resolution[0], prm_rgb_resolution[1])
        
        self.DS_FACTOR   = prm_pcl_processing["pcl_downsample"]
        self.Z_CUTOFF    = prm_pcl_processing["z_cutoff"]
        self.binning_res = prm_pcl_processing["binning_res"]
        self.z_range     = prm_pcl_processing["z_range"]
        self.z_bins      = prm_pcl_processing["z_bins"]    
        
    def _cllb_pointcloud(self, msg):
        """ this callback takes the raw bytestring that is the pointcloud2 message, reshapes it and dumps it into a numpy structured array. This way the bytestring is efficiently decoded. (~0.05ms) """

        # extract the metadata that comes with the PointCloud2 message
        width        = msg.width         # resolution width  (expected: 640)
        height       = msg.height        # resolution height (expected: 480)
        point_step   = msg.point_step    # bytes per point
        row_step     = msg.row_step      # bytes per row
        total_bytes  = len(msg.data)     # total binary size
        is_bigendian = msg.is_bigendian  # endianness flag from ROS
        fields       = msg.fields        # all the data fields and their format (important for decoding!)
        data         = msg.data          # bytestring data 
        
        # define endianness (auto-detect from ROS msg.is_bigendian). numpy: big-endian ('>') / little-endian ('<')
        endianness = '>' if is_bigendian else '<'

        # define a structured dtype to match the 19-byte format specified in "msg.fields"
        point_dtype = np.dtype([
            ('x',         endianness + 'f4'),  # float32 at offset 0
            ('y',         endianness + 'f4'),  # float32 at offset 4
            ('z',         endianness + 'f4'),  # float32 at offset 8
            ('noise',     endianness + 'f4'),  # float32 at offset 12
            ('intensity', endianness + 'u2'),  # uint16  at offset 16
            ('gray',      endianness + 'u1')   # uint8   at offset 18
        ])

        # interpret raw bytes as structured numpy array
        point_cloud = np.frombuffer(data, dtype=point_dtype)
        
        # only take the z values, reshapes them and buffers the message
        self.buffer_pcl      = point_cloud[["x", "y", "z"]]
        self.buffer_pcl_flag = True   
             
        if self.TOF_RES is None:
            self.TOF_RES = (width, height)
    
    def _cllb_uavstate(self, msg):
        """take: data.motors[6].position for cam arm angle position!"""    
        self.buffer_armangle = msg.motors[6].position
     
    def _determine_tof_intrinsics(self, P: np.ndarray, res: tuple):
        """ P is a pointcloud from the TOF with dims (n, 3) where the points have coordinate x, y, z in 3D. resolution is given as a tuple of (width, height) """
        
        # this is the resolution that the tof depht image is expected to be in (for reshaping)
        w = res[0]
        h = res[1]
        
        PX = P[:, 0].reshape(h, w)
        PY = P[:, 1].reshape(h, w)
        PZ = P[:, 2].reshape(h, w)
        
        # normalizing all the points to transform them to the normalized image plane -> lots of measurement points!
        X = PX / PZ
        Y = PY / PZ
        
        # since each columns is supposed to have the same Xn coordinate and each row should have the same Yn coordinate, the average can be taken respectivel. NaN values are ignored. Since there can be some columns / rows with ONLY NaN values, np.nanmean can throw an error but it can safely be ignored, since we'll interpolate full NaN rows / columns later
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xu_lookup = np.nanmean(X, axis=0)
            yv_lookup = np.nanmean(Y, axis=1)
        
        # interpolate missing values with linear regression (this is equivalent to saing the tof cam is distortion free, and the linear interpolation is nothing else but the camera intrinsics matrix that maps pixel coordinates to normalized image plane coordinates. the TOF actually HAS slight distortion at the outermost edge but it's negligible, for a camera with heavier distortion this approach would be wrong!). This is essentially like camera calibration but super easy because we literally know some real-world (3D points) and where they map to in (u, v) image coordinates.
        x_mask    = ~np.isnan(xu_lookup)  
        y_mask    = ~np.isnan(yv_lookup)
        m1, b1    = np.polyfit(np.arange(len(xu_lookup))[x_mask], xu_lookup[x_mask], deg=1)  
        m2, b2    = np.polyfit(np.arange(len(yv_lookup))[y_mask], yv_lookup[y_mask], deg=1) 
        
        # build the H (camera intrinsics) matrix from the interpolation parameters. 
        H_INV = np.array([
            [m1, 0, b1],
            [0, m2, b2],
            [0,  0,  1],
        ])
        H = np.linalg.inv(H_INV)
        
        return H, H_INV

    def _eval_intrinsics_lookup(self, TOF_H_INV: np.ndarray, orig_res: tuple, target_res: tuple):
        # calculate the actual factor that is used to scale the depth image. since the scaled resolutions have to be rounded, it is possible that the actual factor is slightly different from the true one. Since the intrinsics matrix H_INV takes the idx as argument to a linear function basically, the start and end have to match up. start is idx=0 and end is idx=resolution-1, for this reason there is the -1 to determine the actual scaling factor to make both linear ranges match up perfectly.
        x_factor = (target_res[0]-1) / (orig_res[0]-1)
        y_factor = (target_res[1]-1) / (orig_res[1]-1)
        
        # piece together the scaled matrix (only scale the slope, not the intercept)
        H_INV_SCALED = np.array([
            [TOF_H_INV[0, 0] / x_factor,                          0, TOF_H_INV[0, 2]],
            [                         0, TOF_H_INV[1, 1] / y_factor, TOF_H_INV[1, 2]],
            [                         0,                          0,               1],
        ])
        
        # evaluate the linear function (intrinsics matrix) at all the pixel values once
        x_pixels = np.zeros(shape=(3, target_res[0]))
        x_pixels[0, :] = np.arange(target_res[0])
        x_pixels[2, :] = 1
        
        y_pixels = np.zeros(shape=(3, target_res[1]))
        y_pixels[1, :] = np.arange(target_res[1])
        y_pixels[2, :] = 1
        
        xu_lookup_interp = (H_INV_SCALED @ x_pixels)[0, :]
        yv_lookup_interp = (H_INV_SCALED @ y_pixels)[1, :]
        
        return xu_lookup_interp, yv_lookup_interp

    def _process_points(self, pcl_z, xu_lookup_tof, yv_lookup_tof, TOF_RES, TOF_RES_SCALED, PBIN, ZCUTOFF, TF):
        # downsizing Z only ============================================================================================
        # the main benefit here, is that we do not work with the x and y parts because their information is really easy to reconstruct afterwards from kinda the "tof cam intrinsic". this saves a lot of compute.
        PZ = pcl_z.reshape(TOF_RES[1], TOF_RES[0])
        PZ = cv2.resize(PZ, (TOF_RES_SCALED[0], TOF_RES_SCALED[1]))

        # reconstruct full 3d points and apply homogenous transformation ===============================================
        # by tiling the precalculated Xn and Yn normalized image coords we can efficiently reconstruct the full 3D resized pointcloud. Then, a homogenous transformation can be applied to the real 3D points to transform to a different frame: (X, Y, Z, 1) coordinates
        XYZ = np.empty((TOF_RES_SCALED[1], TOF_RES_SCALED[0], 4)) # all points organied in a H x W array
        XYZ[:, :, 0] = np.tile(xu_lookup_tof[None, :], (TOF_RES_SCALED[1], 1)) * PZ # x_n = X/Z -> X = x_n*Z
        XYZ[:, :, 1] = np.tile(yv_lookup_tof[:, None], (1, TOF_RES_SCALED[0])) * PZ # y_n = Y/Z -> Y = y_n*Z
        XYZ[:, :, 2] = PZ
        XYZ[:, :, 3] = 1
        # multiply each [x,y,z,1] point with the T matrix and ditch the homogenous related 1 values (-> batched matrix mult)
        XYZ_TF = np.einsum("ij, HWj -> HWi", TF, XYZ)[:, :, [0, 1, 2]] 

        # project points into normalized image plane again =============================================================
        # through the homogenous transformation, the nice raster structure that the tof intrinsically provides is lost. now the projected points fall onto arbitrary positions on the normalized image plane and work like any other image generating projection again.
        XYZ_TF = XYZ_TF.reshape(-1, 3) # flattening because grid structure is useless anyways now
        XYZ_TF[:, 0] = XYZ_TF[:, 0] / XYZ_TF[:, 2]
        XYZ_TF[:, 1] = XYZ_TF[:, 1] / XYZ_TF[:, 2]
        
        # remove points that have NaN Z depth value ====================================================================
        # XYZ_TF = XYZ_TF[np.isfinite(XYZ_TF[:, 2])]
        XYZ_TF = XYZ_TF[~np.isnan(XYZ_TF[:, 2])]
        
        # remove points that are too close =============================================================================
        # NOTE: up until now, all points have been transformed, even the ones that originated very close to the tof camera. This Z cutoff is now applied only to the NEW coordinate frame. in some cases it might be desired to cutoff the points w.r.t. the tof coordinate frame. in such a case, just apply the cutoff BEFORE the homogenous transformation
        XYZ_TF = XYZ_TF[XYZ_TF[:, 2] > ZCUTOFF]
        
        # binning the projected points =================================================================================
        # since a depth image is the most useful mode to work with down the line, the projected points have to be binned again to kind of simulate pixels and recover a structured representation again. this also allows for patching holes down the line. bins are defined by their edges, also they need to extend by half a cell to mimick how a camera pixel actually does the binning normally. First, tho, the square root is taken of all the z vlaues which has the effect of reducing the binning precision on further away points!
        bins = [
            np.linspace(PBIN["x_range"][0] - PBIN["dx"]/2, PBIN["x_range"][1] + PBIN["dx"]/2, PBIN["x_bins"]+1), # x bins
            np.linspace(PBIN["y_range"][0] - PBIN["dy"]/2, PBIN["y_range"][1] + PBIN["dy"]/2, PBIN["y_bins"]+1), # y bins
            np.linspace(PBIN["z_range"][0] - PBIN["dz"]/2, PBIN["z_range"][1] + PBIN["dz"]/2, PBIN["z_bins"]+1), # z bins
        ]
        xu_lookup_map = bins[0]
        yv_lookup_map = bins[1]

        XYZ_TF[:, 2] = XYZ_TF[:, 2]**0.5 # for better binning precision @ large values
        H, _ = np.histogramdd(XYZ_TF, bins=bins) # watch out, this is now kind of an image, [x<->width, y<->height, z]
    
        # take the weighted average over all the z direction bins. note that the "values" that are assigned to every zbin is basically the midpoint of it's edges! Calculates n_count * value for each bin, sums this up and then divides the sum by the sum of the total binning count for this (x,y) pixel.
        bin_counts   = np.sum(H, axis=2)
        bin_value    = np.linspace(PBIN["z_range"][0]+PBIN["dz"]/2, PBIN["z_range"][1]-PBIN["dz"]/2, PBIN["z_bins"])**2
        weighted_sum = np.einsum("xyz, z -> xy", H, bin_value)
        weighted_avg = np.full(shape=bin_counts.shape, fill_value=np.nan)
        weighted_avg = np.where(bin_counts > 0, weighted_sum/(bin_counts + 1e-6), weighted_avg) # avoid div by 0
        
        # output reordering ============================================================================================
        depth_map = weighted_avg.T # transposing because x(=width) should be the second array dimension in images
        
        return depth_map, xu_lookup_map, yv_lookup_map

    def _inpaint_smooth_defensive(self, depth_map, PBIN):
        """ properly deals with nan values first and especially inpaints first, so that the smoothing only has to deal with sanitized arrays """
        
        # create a mask that is 1 at all the NaNs
        mask = (np.isnan(depth_map)).astype(np.uint8)
        
        # replace nan with 0 for uint conversion
        depth_map_clean = np.where(np.isnan(depth_map), 0, depth_map) 
        
        # normalize and convert image to uint16 (is required by cv2 inpaint, uint16 for a bit more resolution). Watch out, zrange was square-rooted for better binning precision before!
        depth_map_norm = ((depth_map_clean / PBIN["z_range"][1]**2) * 65_535).astype(np.uint16)
        
        # inpaint only the NaN values
        depth_map_inpaint = cv2.inpaint(depth_map_norm, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        
        # denormalize (was only necessary for the uint16 conversion)
        depth_map_inpaint = (depth_map_inpaint / 65_535) * PBIN["z_range"][1]**2
        
        # finally apply a smoothing filter (bilateral conserves edges better)
        depth_map_inpaint = cv2.bilateralFilter(depth_map_inpaint.astype(np.float32), d=5, sigmaColor=30, sigmaSpace=30)
        
        return depth_map_inpaint

    def _iterative_postprocess(self, depth_map, PBIN, diff_thresh):
        """ first smoothes the raw reprojected depth map once and looks at the difference between the smoothed map and the original one, to determine hightly noisy points. These points are then removed and the map is smoothed a second time to achieve good robust inpainting. """
        
        # first iteration smoothing and inpainting (includes noisy points)
        depth_map_it1 = self._inpaint_smooth_defensive(depth_map, PBIN)
        
        # create a mask where the original pixels deviate a lot from the smoothed map
        outlier_mask                = np.abs(depth_map - depth_map_it1) > diff_thresh
        depth_map_it2               = copy.deepcopy(depth_map)
        depth_map_it2[outlier_mask] = np.nan
        
        # second iteration smoothing and inpainting, now without the outliers
        depth_map_fixed = self._inpaint_smooth_defensive(depth_map_it2, PBIN)
        
        return depth_map_fixed

    def _process_pointcloud(self):
        if self.buffer_pcl is None: 
            return
        
        if self.buffer_pcl_flag is False:
            return
        
        if self.TOF_RES is None:
            return
        
        if self.buffer_armangle is None:
            return
            
        # do depth map processing of all the above conditions are met ==================================================
        # reset flag as soon as possible to indicate that this pcl has been processed and no new pcl is missed ---------
        self.buffer_pcl_flag = False
        
        # do the initializing constant calculations once ---------------------------------------------------------------
        if self.do_init_flg is True:
            # once at the beginning automaticall determine the tof intrinsics to efficiently process the tof pcl for all the subequent proccessing steps
            self.do_init_flg = False
            
            pcl = np.concatenate([ # (n, 3) shaped pointloud but now as a normal ndarray
                self.buffer_pcl["x"][:, None], 
                self.buffer_pcl["y"][:, None], 
                self.buffer_pcl["z"][:, None]
                ], axis=1)
            
            self.TOF_H          = self._determine_tof_intrinsics(pcl, res=self.TOF_RES)[0]
            self.TOF_H_INV      = np.linalg.inv(self.TOF_H)
            self.TOF_RES_SCALED = (round(self.DS_FACTOR*self.TOF_RES[0]), round(self.DS_FACTOR*self.TOF_RES[1]))
            self.TOF_XU_LOOKUP  = self._eval_intrinsics_lookup(self.TOF_H_INV, self.TOF_RES, self.TOF_RES_SCALED)[0]
            self.TOF_YV_LOOKUP  = self._eval_intrinsics_lookup(self.TOF_H_INV, self.TOF_RES, self.TOF_RES_SCALED)[1]

        # do the processing --------------------------------------------------------------------------------------------
        depth_map, xu_lookup_map, yv_lookup_map = self._process_points(
            pcl_z           = self.buffer_pcl["z"], 
            xu_lookup_tof         = self.TOF_XU_LOOKUP, 
            yv_lookup_tof         = self.TOF_YV_LOOKUP, 
            TOF_RES        = self.TOF_RES, 
            TOF_RES_SCALED = self.TOF_RES_SCALED, 
            PBIN           = self.BIN_PRM, 
            ZCUTOFF        = self.Z_CUTOFF,
            TF             = (self.TF_IMU2CAM(self.buffer_armangle) @ self.TF_TOF2IMU)
        )
        depth_map_fixed           = self._iterative_postprocess(depth_map, self.BIN_PRM, diff_thresh=0.1)

        # create a color(map) image from the 1-channel depth map
        depth_map_colored         = ((depth_map_fixed / 10) * 255).astype(np.uint8) # 10m is maximum range
        depth_map_colored         = cv2.applyColorMap(depth_map_colored, cv2.COLORMAP_TURBO)
        
        # compress and publish -----------------------------------------------------------------------------------------
        # publish the full depth map information        
        xu_lookup_map_msg         = Float32MultiArray()
        xu_lookup_map_msg.data    = xu_lookup_map.tolist()
        yv_lookup_map_msg         = Float32MultiArray()
        yv_lookup_map_msg.data    = yv_lookup_map.tolist()
        depthmap_msg              = DepthMap() 
        depthmap_msg.header.stamp = rospy.Time.now()
        depthmap_msg.depthmap     = CvBridge().cv2_to_imgmsg(depth_map_fixed, encoding="32FC1")
        depthmap_msg.xu_lookup    = xu_lookup_map_msg
        depthmap_msg.yv_lookup    = yv_lookup_map_msg
        self.PubDepthMap.publish(depthmap_msg)
        
        # publish the artificially colored depth map debug image
        img_msg = Bridge.cv2_to_compressed_imgmsg(depth_map_colored, dst_format = "jpg")
        self.PubImg.publish(img_msg)
        
    def _startup_log(self):
        
        param_list = [
            dict(name = "run_hz",  value = self.run_hz,  suffix = "Hz"),
            
            dict(name = "f_hat_x", value = self.f_hat_x, suffix = None),
            dict(name = "f_hat_y", value = self.f_hat_y, suffix = None),
            dict(name = "cx",      value = self.cx,      suffix = None),
            dict(name = "cy",      value = self.cy,      suffix = None),
            
            dict(name = "cam_res", value = self.CAM_RES, suffix = None),
            
            dict(name = "ds_factor",   value = self.DS_FACTOR,   suffix = None),
            dict(name = "z_cutoff",    value = self.Z_CUTOFF,    suffix = None),
            dict(name = "binning_res", value = self.binning_res, suffix = None),
            dict(name = "z_range",     value = self.z_range,     suffix = None),
            dict(name = "z_bins",      value = self.z_bins,      suffix = None),
        ]
        
        rospy.loginfo(generic_startup_log("Depth Map Fusion", param_list, column_width = 80))
        
    def _RUN(self):
        self._startup_log()
        while not rospy.is_shutdown():
            self._process_pointcloud()
            self.Rate.sleep()
        
    
if __name__ == '__main__':
    try:
        node = NodeDepthMap()
    except rospy.ROSInterruptException:
        pass
