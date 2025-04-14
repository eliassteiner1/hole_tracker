cmake_minimum_required(VERSION 3.0.2)
project(hole_tracker)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)


################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  DetectionPoints.msg
  DepthMap.msg
  )

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################


###################################
## catkin specific configuration ##
###################################

catkin_package(
   CATKIN_DEPENDS roscpp rospy std_msgs
   DEPENDS message_runtime
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)


#############
## Install ##
#############


#############
## Testing ##
#############


