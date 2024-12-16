#!/bin/bash

# 获取当前脚本所在路径
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 找到工作空间的根目录（假设是包含 install 文件夹的上级目录）
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR" && while [ ! -d install ]; do cd ..; done; pwd)

# 打印当前工作空间路径
echo "Workspace root: $WORKSPACE_ROOT"

# 切换到工作空间根目录
cd "$WORKSPACE_ROOT" || {
    echo "Error: Failed to navigate to workspace root"
    exit 1
}

# Source ROS2 的 setup 文件
echo "Sourcing ROS2 workspace..."
source install/setup.bash

# 启动 ROS2 launch 文件
echo "Launching UR5 visualization..."
ros2 launch ur5_arm_zell_description ur5_arm_visualize.launch.py
