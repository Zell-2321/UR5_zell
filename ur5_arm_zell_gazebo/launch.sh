#!/usr/bin/env bash

#### TODO: 

# # 获取当前脚本所在路径
# SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# # 找到工作空间的根目录（假设是包含 install 文件夹的上级目录）
# WORKSPACE_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d install ]; do cd ..; done; pwd)

# # 打印当前工作空间路径
# echo "Workspace root: $WORKSPACE_ROOT"

# # 切换到工作空间根目录
# cd "$WORKSPACE_ROOT" || {
#     echo "Error: Failed to navigate to workspace root"
#     exit 1
# }

# echo "找到ROS2工作空间路径: $WORKSPACE_ROOT"
# sleep 2

# source /opt/ros/humble/setup.bash
# source install/setup.bash

# # 使用 gnome-terminal 命令在新窗口中执行 ros2 launch
# # gnome-terminal --window -- bash --login -c "
# #     cd ${WORKSPACE_ROOT} && \
# #     ros2 launch ur5_arm_zell_gazebo start_world.launch.py; \
# #     exec bash
# # "
# # sleep 10

# # 再启动一个新窗口（或标签），source 工作区环境
# gnome-terminal --tab -- bash --login -c "
#     cd ${WORKSPACE_ROOT} ; \
#     ros2 launch ur5_arm_zell_gazebo spawn_robot_ros2.launch.xml ; \
#     exec bash
# "
