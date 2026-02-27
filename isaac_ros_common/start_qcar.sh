#!/bin/bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/disable_shm.xml
ros2 launch qcar2_nodes qcar2_virtual_launch.py
