# UR5 Xacro Files Overview / UR5 Xacro 文件说明

## Introduction / 简介
This document provides an overview of the Xacro files used for defining and simulating the UR5 robotic arm in a ROS2 and Gazebo environment. The modular structure ensures flexibility and ease of modification.

本文档概述了用于 ROS2 和 Gazebo 环境中定义和模拟 UR5 机器臂的 Xacro 文件。该模块化结构确保了添加和修改的便捷性。

---

## File Descriptions / 文件描述

### 1. **ur5_main.urdf.xacro**
**Description:** This is the main file that instantiates the UR5 robotic arm. It includes:
- Definitions for `ros2_control` interfaces.
- Relevant Gazebo plugins for simulation.

**说明：** 此文件为主文件，实例化 UR5 机器臂。包含：
- `ros2_control`控制接口定义
- Gazebo 模拟相关插件

---

### 2. **ur5_gazebo.urdf.xacro**
**Description:** A template file for defining Gazebo-related plugins.

**说明：** 此文件为定义 Gazebo 相关插件的模板文件。

---

### 3. **ur5_parameters.urdf.xacro**
**Description:** A template file containing all the parameter definitions. This file should always be loaded first in instantiation files to ensure parameter priority.

**说明：** 此文件包含所有参数定义。在实例化文件中，应优先加载。

---

### 4. **ur5_structure.urdf.xacro**
**Description:** A template file for defining the structural elements of the robotic arm, including sensor integrations.

**说明：** 此文件定义机器臂的结构元素，包括传感器的集成。

---

### 5. **ur5_control.urdf.xacro**
**Description:** A template file defining controllers associated with the UR5 robotic arm.

**说明：** 此文件定义 UR5 机器臂相关控制器。

---

## Usage Guidelines / 使用指南
1. Always load `ur5_parameters.urdf.xacro` first to ensure correct parameter initialization.
2. Use `ur5_main.urdf.xacro` as the entry point for instantiating the full robot model.
3. Modify template files (`ur5_gazebo`, `ur5_structure`, `ur5_control`) as needed for custom configurations.

1. 优先加载 `ur5_parameters.urdf.xacro` 以确保参数正确初始化。
2. 使用 `ur5_main.urdf.xacro` 作为全机型实例化的入口。
3. 根据需要修改模板文件（`ur5_gazebo` ，`ur5_structure` ，`ur5_control`）实现自定义配置。

---

## File Hierarchy / 文件库结构
```
.
├── ur5_main.urdf.xacro         # Main instantiation file
├── ur5_parameters.urdf.xacro   # Parameter definitions
├── ur5_gazebo.urdf.xacro       # Gazebo plugin definitions
├── ur5_structure.urdf.xacro    # Structural definitions
└── ur5_control.urdf.xacro      # Controller definitions
```

---

## Notes / 注意事项
- Ensure all Xacro files are properly included in your ROS2 workspace.
- Use `ros2 launch` to test the configuration in Gazebo.
- Validate URDF outputs using tools like `check_urdf` or `urdf_parser`.

- 确保所有 Xacro 文件被正确加载到 ROS2 工作区。
- 使用 `ros2 launch` 在 Gazebo 中测试配置。
- 使用工具（如 `check_urdf` 或 `urdf_parser`）验证 URDF 输出。

