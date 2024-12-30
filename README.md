# UR5_zell
UR5 robotics arm simulation project

# packages
- ur5_arm_zell_description: UR5 robotic arm description package
- ur5_arm_zell_gazebo: UR5 robotic arm simulation package
- TODO: ur5_arm_zell_bringup: UR5 robotic arm bringup package

# README - ROS Package Overview

## 项目概述
此项目基于ROS (Robot Operating System) 框架，包含多个ROS包，每个包负责不同的功能模块。以下是各个ROS包的功能描述及其基本格式。

## 包结构说明
每个ROS包遵循以下基本格式：
```
<package_name>/
|-- CMakeLists.txt           # 编译规则
|-- package.xml              # 包的元信息
|-- launch/                  # 启动文件目录
|   |-- <launch_file>.launch # 启动文件
|-- src/                     # 源代码目录
|   |-- <source_code>.cpp    # 源文件
|-- include/<package_name>/  # 头文件目录
|   |-- <header>.h           # 头文件
|-- config/                  # 配置文件目录
|   |-- <config_file>.yaml   # 配置文件
|-- msg/                     # 自定义消息目录
|   |-- <message>.msg        # 消息定义文件
|-- srv/                     # 自定义服务目录
|   |-- <service>.srv        # 服务定义文件
|-- scripts/                 # Python脚本目录
|   |-- <script>.py          # 脚本文件
```

## ROS包功能概述

## TODO:
## example:
<!-- ### 1. robot_controller
**功能描述：**
管理机器人运动控制，包括路径规划与执行。
- **话题：** `/cmd_vel` - 机器人的速度控制指令。
- **服务：** `/set_goal` - 设定目标位置。
- **消息类型：** `geometry_msgs/Twist`

### 2. sensor_interface
**功能描述：**
传感器数据采集与发布，包括激光雷达与IMU数据。
- **话题：** `/scan` - 雷达扫描数据。
- **服务：** `/calibrate_imu` - IMU传感器校准。
- **消息类型：** `sensor_msgs/LaserScan`

### 3. navigation_planner
**功能描述：**
全局与局部路径规划，实现机器人导航。
- **话题：** `/map` - 环境地图数据。
- **服务：** `/plan_path` - 规划路径。
- **消息类型：** `nav_msgs/Path`

### 4. object_detection
**功能描述：**
视觉检测模块，识别环境中的物体。
- **话题：** `/detected_objects` - 物体检测结果。
- **服务：** `/detect` - 请求物体检测。
- **消息类型：** `sensor_msgs/Image` -->

<!-- ## 使用方法
1. **编译包：**
```bash
catkin_make
```
2. **运行节点：**
```bash
roslaunch <package_name> <launch_file>.launch
```
3. **检查话题和服务：**
```bash
rostopic list
rosservice list
```

## 依赖关系
- ROS版本: Humble
- 必要依赖包: `roscpp`, `std_msgs`, `sensor_msgs`, `nav_msgs`

## 联系方式
如有问题，请联系：`ros_support@example.com`。 -->


