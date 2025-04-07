#!/usr/bin/env python3
import os, sys, time
import warnings
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt

import rospy
from   sensor_msgs.msg   import PointCloud2
from   sensor_msgs.msg   import CompressedImage

from   utils.image_tools import ImageTools


# Mapping of ROS PointField data types to NumPy dtype codes
ROS_TO_NUMPY_DTYPE = {
    1:  'i1',  # int8     → 'i1'  (1-byte signed integer)
    2:  'u1',  # uint8    → 'u1'  (1-byte unsigned integer)
    3:  'i2',  # int16    → 'i2'  (2-byte signed integer)
    4:  'u2',  # uint16   → 'u2'  (2-byte unsigned integer)
    5:  'i4',  # int32    → 'i4'  (4-byte signed integer)
    6:  'u4',  # uint32   → 'u4'  (4-byte unsigned integer)
    7:  'f4',  # float32  → 'f4'  (4-byte floating point)
    8:  'f8'   # float64  → 'f8'  (8-byte floating point)
}

def determine_tof_intrinsics(P: np.ndarray, res: tuple):
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
        rx = np.nanmean(X, axis=0)
        ry = np.nanmean(Y, axis=1)
    
    # interpolate missing values with linear regression (this is equivalent to saing the tof cam is distortion free, and the linear interpolation is nothing else but the camera intrinsics matrix that maps pixel coordinates to normalized image plane coordinates. the TOF actually HAS slight distortion at the outermost edge but it's negligible, for a camera with heavier distortion this approach would be wrong!). This is essentially like camera calibration but super easy because we literally know some real-world (3D points) and where they map to in (u, v) image coordinates.
    x_mask    = ~np.isnan(rx)  
    y_mask    = ~np.isnan(ry)
    m1, b1    = np.polyfit(np.arange(len(rx))[x_mask], rx[x_mask], deg=1)  
    m2, b2    = np.polyfit(np.arange(len(ry))[y_mask], ry[y_mask], deg=1) 
    
    # build the H (camera intrinsics) matrix from the interpolation parameters. 
    H_INV = np.array([
        [m1, 0, b1],
        [0, m2, b2],
        [0,  0,  1],
    ])
    H = np.linalg.inv(H_INV)
    
    return H, H_INV

def evaluate_res_vecs(TOF_H_INV: np.ndarray, orig_res: tuple, target_res: tuple):
    
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
    
    rx_interp = (H_INV_SCALED @ x_pixels)[0, :]
    ry_interp = (H_INV_SCALED @ y_pixels)[1, :]
    
    return rx_interp, ry_interp

def process_points(pclZ, rx, ry, TOF_RES, TOF_RES_SCALED, PBIN, ZCUTOFF, TF):
    # downsizing Z only ================================================================================================
    # the main benefit here, is that we do not work with the x and y parts because their information is really easy to reconstruct afterwards from kinda the "tof cam intrinsic". this saves a lot of compute.
    PZ = pclZ.reshape(TOF_RES[1], TOF_RES[0])
    PZ = cv2.resize(PZ, (TOF_RES_SCALED[0], TOF_RES_SCALED[1]))

    # reconstruct full 3d points and apply homogenous transformation ===================================================
    # by tiling the precalculated Xn and Yn normalized image coords we can efficiently reconstruct the full 3D resized pointcloud. Then, a homogenous transformation can be applied to the real 3D points to transform to a different frame: (X, Y, Z, 1) coordinates
    XYZ = np.empty((TOF_RES_SCALED[1], TOF_RES_SCALED[0], 4)) # all points organied in a H x W array
    XYZ[:, :, 0] = np.tile(rx[None, :], (TOF_RES_SCALED[1], 1)) * PZ # x_n = X/Z -> X = x_n*Z
    XYZ[:, :, 1] = np.tile(ry[:, None], (1, TOF_RES_SCALED[0])) * PZ # y_n = Y/Z -> Y = y_n*Z
    XYZ[:, :, 2] = PZ
    XYZ[:, :, 3] = 1
    # multiply each [x,y,z,1] point with the T matrix and ditch the homogenous related 1 values (-> batched matrix mult)
    XYZ_TF = np.einsum("ij, HWj -> HWi", TF, XYZ)[:, :, [0, 1, 2]] 

    # project points into normalized image plane again =================================================================
    # through the homogenous transformation, the nice raster structure that the tof intrinsically provides is lost. now the projected points fall onto arbitrary positions on the normalized image plane and work like any other image generating projection again.
    XYZ_TF = XYZ_TF.reshape(-1, 3) # flattening because grid structure is useless anyways now
    XYZ_TF[:, 0] = XYZ_TF[:, 0] / XYZ_TF[:, 2]
    XYZ_TF[:, 1] = XYZ_TF[:, 1] / XYZ_TF[:, 2]
    
    # remove points that have NaN Z depth value ========================================================================
    # XYZ_TF = XYZ_TF[np.isfinite(XYZ_TF[:, 2])]
    XYZ_TF = XYZ_TF[~np.isnan(XYZ_TF[:, 2])]
    
    # remove points that are too close =================================================================================
    # NOTE: up until now, all points have been transformed, even the ones that originated very close to the tof camera. This Z cutoff is now applied only to the NEW coordinate frame. in some cases it might be desired to cutoff the points w.r.t. the tof coordinate frame. in such a case, just apply the cutoff BEFORE the homogenous transformation
    XYZ_TF = XYZ_TF[XYZ_TF[:, 2] > ZCUTOFF]
    
    # binning the projected points =====================================================================================
    # since a depth image is the most useful mode to work with down the line, the projected points have to be binned again to kind of simulate pixels and recover a structured representation again. this also allows for patching holes down the line. bins are defined by their edges, also they need to extend by half a cell to mimick how a camera pixel actually does the binning normally. First, tho, the square root is taken of all the z vlaues which has the effect of reducing the binning precision on further away points!
    bins = [
        np.linspace(PBIN["x_range"][0] - PBIN["dx"]/2, PBIN["x_range"][1] + PBIN["dx"]/2, PBIN["x_bins"]+1), # x bins
        np.linspace(PBIN["y_range"][0] - PBIN["dy"]/2, PBIN["y_range"][1] + PBIN["dy"]/2, PBIN["y_bins"]+1), # y bins
        np.linspace(PBIN["z_range"][0] - PBIN["dz"]/2, PBIN["z_range"][1] + PBIN["dz"]/2, PBIN["z_bins"]+1), # z bins
    ]
    XYZ_TF[:, 2] = XYZ_TF[:, 2]**0.5 # for better binning precision @ large values
    H, _ = np.histogramdd(XYZ_TF, bins=bins) # watch out, this is now kind of an image, [x<->width, y<->height, z]
  
    # take the weighted average over all the z direction bins. note that the "values" that are assigned to every zbin is basically the midpoint of it's edges! Calculates n_count * value for each bin, sums this up and then divides the sum by the sum of the total binning count for this (x,y) pixel.
    bin_counts   = np.sum(H, axis=2)
    bin_value    = np.linspace(PBIN["z_range"][0]+PBIN["dz"]/2, PBIN["z_range"][1]-PBIN["dz"]/2, PBIN["z_bins"])**2
    weighted_sum = np.einsum("xyz, z -> xy", H, bin_value)
    weighted_avg = np.full(shape=bin_counts.shape, fill_value=np.nan)
    weighted_avg = np.where(bin_counts > 0, weighted_sum/(bin_counts + 1e-6), weighted_avg) # avoid div by 0
    
    # output reordering ================================================================================================
    depth_map = weighted_avg.T # transposing because x(=width) should be the second array dimension in images
    depth_map = depth_map[::-1, ::-1] # flip axes: tof coordinate frame is flipped compared to a conventional img frame

    return depth_map

def inpaint_defensive(depth_map, PBIN):
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

def iterative_smoothing(depth_map, PBIN, diff_thresh):
    """ first smoothes the raw reprojected depth map once and looks at the difference between the smoothed map and the original one, to determine hightly noisy points. These points are then removed and the map is smoothed a second time to achieve good robust inpainting. """
    
    # first iteration smoothing and inpainting (includes noisy points)
    depth_map_it1 = inpaint_defensive(depth_map, PBIN) # TODO: don't call this function "globally"
    
    # create a mask where the original pixels deviate a lot from the smoothed map
    outlier_mask                = np.abs(depth_map - depth_map_it1) > diff_thresh
    depth_map_it2               = copy.deepcopy(depth_map)
    depth_map_it2[outlier_mask] = np.nan
    
    # second iteration smoothing and inpainting, now without the outliers
    depth_map_fixed = inpaint_defensive(depth_map_it2, PBIN) # TODO: don't call this function "globally"
    
    return depth_map_fixed

Converter = ImageTools()


class NodeDepthMap:
    
    def __init__(self):
        rospy.init_node("depthmap_node")
        self._get_params()
        
        self.Rate = rospy.Rate(self.run_hz)
        
        # [set some processing params] =================================================================================
        self.DO_INIT_FLG = True
        
        self.TOF_H     = None # determine automatically once
        self.TOF_H_INV = None # determine automatically once
        self.TOF_RES   = None # determine automatically once (W, H)
        
        self.CAM_H     = np.array([
            [1210.19, 0000.00, 0717.19],
            [0000.00, 1211.66, 0486.47],
            [0000.00, 0000.00, 0001.00],
        ])
        self.CAM_H_INV = np.linalg.inv(self.CAM_H)
        self.CAM_RES   = (1440, 1080) # (W, H)
        
        self.DS_FACTOR = 0.2 # for pcl downsampling
        self.Z_CUTOFF  = 0.4 # this applies AFTER the homogenous transformation
        
        self.TF_TOF2CAM = np.array([ # TODO: replace with the lambdified correct tf!
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.PMIN = (self.CAM_H_INV @ np.array([0, 0, 1])[:, None]).squeeze()
        self.PMAX = (self.CAM_H_INV @ np.array([self.CAM_RES[0]-1, self.CAM_RES[1]-1, 1])[:, None]).squeeze()
        self.BINNING_RES = 100
        self.BIN_PRM = dict(
            x_range = [self.PMIN[0], self.PMAX[0]],
            y_range = [self.PMIN[1], self.PMAX[1]],
            z_range = [0.0, 10.0], # z_range[0] has to be >= 0!
            
            x_bins  = round((self.PMAX[0] - self.PMIN[0]) * self.BINNING_RES),
            y_bins  = round((self.PMAX[1] - self.PMIN[1]) * self.BINNING_RES),
            z_bins  = 128,
            
            dx      = None,
            dy      = None,
            dz      = None,
        )
        self.BIN_PRM["z_range"][1] = (self.BIN_PRM["z_range"][1]**0.5) # x² scaled for better precision
        self.BIN_PRM["dx"] = (self.BIN_PRM["x_range"][1] - self.BIN_PRM["x_range"][0]) / self.BIN_PRM["x_bins"]
        self.BIN_PRM["dy"] = (self.BIN_PRM["y_range"][1] - self.BIN_PRM["y_range"][0]) / self.BIN_PRM["y_bins"]
        self.BIN_PRM["dz"] = (self.BIN_PRM["z_range"][1] - self.BIN_PRM["z_range"][0]) / self.BIN_PRM["z_bins"]
        
        self.TOF_RES_SCALED = None # determine when TOF resolution has been determined automatically (W, H)
        
        self.RX = None # determine from tof Intrinsics
        self.RY = None # determine from tof Intrinsics
        # ==============================================================================================================

        # for input buffering
        self.buffer_pcl      = None # (H*W) x [X, Y, Z]
        self.buffer_pcl_flag = False

        # setup publishers and subscribers last, so that all needed callback methods are defined already
        # TODO: needs subscribing to arm angle
        self.SubPcl = rospy.Subscriber("input_pcl", PointCloud2, self._cllb_pointcloud, queue_size=1)
        self.PubDepthMap = rospy.Publisher("output_depthmap", CompressedImage, queue_size=1)
        
        self._run()
        
    def _get_params(self):
        self.run_hz = rospy.get_param("~run_hz", 10.0)
        
    def _cllb_pointcloud(self, msg):
        """ this callback takes the raw bytestring that is the pointcloud2 message, reshapes it and dumps it into a numpy structured array. This way the bytestring is efficiently decoded. (~0.05ms) """
        
        t0 = time.perf_counter()
        
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

        
        t1 = time.perf_counter()
        print(f"decoding took: {(t1-t0)*1000}ms!")
    
    def _cllb_uavstate(self, msg):
        
        ...
            
    def _process_pointcloud(self):
        if (self.buffer_pcl is not None) and (self.TOF_RES is not None) and (self.buffer_pcl_flag is True):
            self.buffer_pcl_flag = False
            
            print("doing work! ============================= ")
            
            # do the initializing constant calculations once ===========================================================
            if self.DO_INIT_FLG is True:
                self.DO_INIT_FLG = False
                
                pcl = np.concatenate([ # (n, 3) shaped pointloud but now as a normal ndarray
                    self.buffer_pcl["x"][:, None], 
                    self.buffer_pcl["y"][:, None], 
                    self.buffer_pcl["z"][:, None]
                    ], axis=1)
                self.TOF_H, self.TOF_H_INV = determine_tof_intrinsics(pcl, res=self.TOF_RES)
                self.TOF_RES_SCALED = (round(self.DS_FACTOR * self.TOF_RES[0]), round(self.DS_FACTOR * self.TOF_RES[1]))
                self.RX, self.RY = evaluate_res_vecs(self.TOF_H_INV, self.TOF_RES, self.TOF_RES_SCALED)
                
            # do the processing ========================================================================================
            depth_map = process_points(
                self.buffer_pcl["z"], 
                self.RX, 
                self.RY, 
                self.TOF_RES, 
                self.TOF_RES_SCALED, 
                self.BIN_PRM, 
                self.Z_CUTOFF,
                self.TF_TOF2CAM
            )

            depth_map_fixed = iterative_smoothing(depth_map, self.BIN_PRM, diff_thresh=0.1)
            
            
            # # just format so that it's accepted as an image for compression
            # depth_map_fixed = np.tile(depth_map_fixed[:, :, None], (1, 1, 3))
            # depth_map_fixed = ((depth_map_fixed / 10) * 255).astype(np.uint8)

            depth_map_fixed = ((depth_map_fixed / 10) * 255).astype(np.uint8) # 10m is maximum range
            depth_map_fixed = cv2.applyColorMap(depth_map_fixed, cv2.COLORMAP_TURBO)
            
            # compress and publish
            depth_map_msg = Converter.convert_cv2_to_ros_compressed_msg(depth_map_fixed, compressed_format="jpeg")
            self.PubDepthMap.publish(depth_map_msg)
            
            
            
            
            
            
            
            
        
    def _startup_log():
        # print params and startup message
        ...
        
    def _run(self):
        while not rospy.is_shutdown():
            self._process_pointcloud()
            self.Rate.sleep()
        
    
if __name__ == '__main__':
    try:
        node = NodeDepthMap()
    except rospy.ROSInterruptException:
        pass
