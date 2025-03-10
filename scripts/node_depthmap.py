#!/usr/bin/env python3
import os, sys, time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import PointCloud2

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

def pointcloud_callback_rawdecode(msg):
    t0 = time.perf_counter()
    
    # Extract metadata from PointCloud2 message
    width        = msg.width         # Expected: 640
    height       = msg.height        # Expected: 480
    point_step   = msg.point_step    # Bytes per point
    row_step     = msg.row_step      # Bytes per row (if organized)
    total_bytes  = len(msg.data)     # Total binary size
    is_bigendian = msg.is_bigendian  # Endianness flag from ROS
    fields       = msg.fields        # all the data fields and their format (IMPORTANT FOR DECODING)
    data         = msg.data          # bytestring data
    
    for f in fields:
        print(f.name, f.datatype)    
    
    # Define endianness (auto-detect from ROS msg.is_bigendian)
    endianness = '>' if is_bigendian else '<'  # Big-endian ('>') or Little-endian ('<')

    # Define a structured dtype to match the 19-byte format specified in "msg.fields"
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
    
    # only take the z values
    PZ = point_cloud["z"]
    PZ = PZ.reshape(height, width)
    
    t1 = time.perf_counter()
    print(f"decoding took: {(t1-t0)*1000}ms!")


def listener():
    rospy.init_node('pointcloud_listener_node', anonymous=True)
    
    # Replace '/pointcloud_topic' with your actual topic name.
    rospy.Subscriber('/quail/normal_estimation/pico_flexx/points', PointCloud2, pointcloud_callback_rawdecode)
    
    rospy.spin()

if __name__ == '__main__':
    listener()
