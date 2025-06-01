# hole_tracker ROS Package for Master Thesis of Elias Steiner
This ROS package is part of the Master Thesis of Elias Steiner conducted with the Autonomous Systems Lab at ETH Zürich. For any details regarding the overarching project, please consult the thesis report ``master_thesis_report.pdf``. 

## Overview
The main aim of this thesis was to perform a reactive peg insertion task with a fully actuated omnidirectional hexacopter (platform developed by ASL). The system involves multiple different modules. A visual Detector to produce raw detections of the target from an RGB image. An Estimator to clean up these points and ultimately output one consistent estimate of the target location. Furthermore, a Controller tasked with steering the drone to the target hole and holding a stable position - even upon contact. Finally, passively compliant end effectors were also tested (see system overview diagram below). 

<p align="center">
  <img src="/overview.png" width="800" alt="system overview diagram" />
</p>

The entire object tracking module (comprised of Detector and Estimator) is implemented in the form of a ROS package (hole_tracker) and is designed to be executed within the existing platform environment; this repository contains all necessary code. Each functional entity is packaged as a separate ROS node, which can all be found in the ``/scripts`` folder. Some repeatedly used functions as well as larger, self-contained functions can be found in the ``/scripts/utils`` folder. All tunable parameters for these nodes can be set from one single yaml file which is located in the ``/config`` folder. Launch files for each node are located in the ``/launch`` folder and mainly handle the remapping of in- and output topics.

## The hole_tracker Package in Detail
The ``/scripts`` folder contains the ROS nodes that were needed for the Tracker module - all explained in the following list.


- The **Blob tracker Detector module** variant is implemented in ``scripts/node_detector_blob.py``. As an input it expects an RGB image and all detected keypoints are output as a custom ``DetectionPoints`` message (see ``/msg`` folder). Optionally a "debug image" can be published, to visualize the detections.

- **The YOLO Detector module** variant is implemented in ``scripts/node_detector_yolo.py``. It also takes an RGB image as input and outputs the same ``DetectionPoints`` message. The "debug image" is also available. 

  However, the YOLO model was run both as a compiled model on the drone (TensorRT) and with the standard Ultralytics API on a laptop for testing. Therefore, custom code was needed for inference, depending on the framework used. ``scripts/utils/multi_framework_yolo.py`` provides a convenience class which can run YOLO detection with both frameworks. While the Ultralytics API is lightweight, the TensorRT inference is more intricate and is located in a separate file: ``scripts/utils/tensorrt_inference.py``. 

  The compiled models, as well as the trained weights for the Ultralytics API are located in the ``/nnets`` folder.

- The ROS wrapper for the **Estimator** module is contained in ``scripts/node_tracker.py``. It primarily handles the real-time execution and buffering of messages. The main Estimator functionality itself is implemented in ``scripts/utils/hole_tracker.py``. Since the Estimator also has to transform image points, it needs access to the camera distortion model which can be found in ``script/utils/equidistant_distorter.py``.

  The following message topics are required by this node:

  - odometry (from drone)
  - uavstate, arm angle (from drone)
  - RGB image (from drone)
  - depth information, either normal estimation (from drone) or sensor fusion (from depthmap node)
  - detection points (from detector)

  This node publishes two topics:

  -  the 3D estimate point
  -  an optional debug visualization image

- Besides the already available normal estimation, **full sensor fusion (ToF and RGB camera)** is implemented as a source of depth information in ``scripts/node_depthmap.py``. As an input topic, the raw point cloud from the ToF camera is needed, and a custom ``DepthMap`` message is published  (see ``/msg`` folder). This message contains the transformed artificial depth image for the RGB image plane.

- For evaulation, the **target estimate point** had to be **transformed** to the "wall" coordinate frame, and all measurements were written to a csv log file. This is handled by ``scripts/node_estimate_tf.py``.
