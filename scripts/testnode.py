#!/usr/bin/env python3

import rospy

def main():
    rospy.init_node("my_node")

    # Retrieve parameters from the ROS parameter server
    robot_speed = rospy.get_param("~robot_speed", 1.0)  # "~" makes it private to this node
    sensor_threshold = rospy.get_param("~sensor_threshold", 0.5)
    use_simulation = rospy.get_param("~use_simulation", True)

    rospy.loginfo(f"Robot Speed: {robot_speed}")
    rospy.loginfo(f"Sensor Threshold: {sensor_threshold}")
    rospy.loginfo(f"Use Simulation: {use_simulation}")

    rospy.spin()

if __name__ == "__main__":
    main()
