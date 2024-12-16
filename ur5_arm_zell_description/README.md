# UR5 Arm Description Package  
# UR5机械臂描述包  

## 简介 | Overview  
该包用于定义 UR5 机械臂的模型描述，包括结构、参数、插件和 Gazebo 仿真配置。  
This package defines the UR5 robotic arm model description, including structure, parameters, plugins, and Gazebo simulation configuration.  

---

## 文件夹结构 | Directory Structure  

```plaintext
.
├── config          # 配置文件，例如控制参数或 YAML 配置  
│                  # Configuration files, e.g., control parameters or YAML configurations  
├── launch          # ROS 2 启动文件，用于启动描述、仿真和其他功能  
│                  # ROS 2 launch files for robot description, simulation, and more  
├── meshes          # 3D 模型文件（STL/DAE 格式），供可视化和碰撞检测使用  
│                  # 3D model files (STL/DAE) for visualization and collision detection  
├── rviz            # RViz 配置文件（如视角、主题设置）  
│                  # RViz configuration files (e.g., views and topic settings)  
├── scripts         # 脚本文件（例如 Python 辅助脚本）  
│                  # Scripts (e.g., Python helper scripts)  
├── src             # 源代码文件（C++ 或 Python 实现）  
│                  # Source code files (C++ or Python implementations)  
├── urdf            # Xacro/URDF 文件，定义机械臂模型及相关插件  
│   ├── ur5_main.urdf.xacro         # 实例化文件：包含所有组件与插件  
│   │                              # Instantiated file: includes all components and plugins  
│   ├── ur5_gazebo.urdf.xacro       # 模板文件：Gazebo 插件定义  
│   │                              # Template file: Gazebo plugin definitions  
│   ├── ur5_parameters.urdf.xacro   # 模板文件：所有参数的定义，优先加载  
│   │                              # Template file: all parameter definitions (loaded first)  
│   ├── ur5_structure.urdf.xacro    # 模板文件：UR5 机械臂结构及传感器定义  
│   │                              # Template file: UR5 arm structure and sensor definitions  
│   └── ur5_control.urdf.xacro      # 模板文件：UR5 机械臂相关控制器定义  
│                                  # Template file: UR5 control configuration using ros2_control  
├── CMakeLists.txt   # 构建系统配置文件 | Build system configuration file  
├── package.xml      # ROS 2 包配置文件 | ROS 2 package configuration file  
└── README.md        # 当前文档 | This document  
```

---

## 主要文件说明 | Key Files Description  

1. **`ur5_main.urdf.xacro`**  
   - 功能：实例化 UR5 机械臂的完整模型。  
     Function: Instantiates the full UR5 robotic arm model.  
   - 包含 | Includes:  
     - 结构定义（`ur5_structure.urdf.xacro`）  
       Structure definition (`ur5_structure.urdf.xacro`)  
     - 参数定义（`ur5_parameters.urdf.xacro`）  
       Parameter definitions (`ur5_parameters.urdf.xacro`)  
     - Gazebo 控制插件（`ur5_gazebo.urdf.xacro`）  
       Gazebo plugin definitions (`ur5_gazebo.urdf.xacro`)  
     - `ros2_control` 插件配置（`ur5_control.urdf.xacro`）  
       `ros2_control` plugin configuration (`ur5_control.urdf.xacro`)  

2. **`ur5_gazebo.urdf.xacro`**  
   - 功能：定义 Gazebo 仿真相关插件。  
     Function: Defines Gazebo simulation-related plugins.  

3. **`ur5_parameters.urdf.xacro`**  
   - 功能：定义 UR5 机械臂所有参数，例如长度、关节属性等。  
     Function: Defines all parameters for UR5, such as lengths and joint properties.  

4. **`ur5_structure.urdf.xacro`**  
   - 功能：定义 UR5 机械臂的结构及传感器。  
     Function: Defines the structure and sensors for the UR5 robotic arm.  

5. **`ur5_control.urdf.xacro`**  
   - 功能：定义 `ros2_control` 插件，用于控制 UR5 机械臂的运动。  
     Function: Configures the `ros2_control` plugin for controlling UR5 robotic arm movement.  

---

## 如何使用 | How to Use  

1. **验证 Xacro 文件 | Validate Xacro Files**  
   将 Xacro 文件转换为 URDF 格式，检查其语法是否正确：  
   Convert the Xacro file to URDF format and check for syntax errors:  
   ```bash
   xacro urdf/ur5_main.urdf.xacro > ur5_robot.urdf
   check_urdf ur5_robot.urdf
   ```

2. **启动机器人模型（RViz2 仿真） | Launch Robot Model in RViz2**  
   使用 `launch` 文件在 RViz2 中可视化模型：  
   Use the `launch` file to visualize the model in RViz2:  
   ```bash
   ros2 launch ur5_arm_zell_description ur5_arm_visualize.launch.py
   ```


---

## 依赖项 | Dependencies  

- **ROS 2**（建议使用 Humble 版本或以上）  
  **ROS 2** (Recommended Humble or later)  
- **xacro**：用于解析和生成 URDF 文件  
  **xacro**: For parsing and generating URDF files  
- **Gazebo**：仿真环境  
  **Gazebo**: Simulation environment  
- **robot_state_publisher**：发布机器人模型状态  
  **robot_state_publisher**: Publishes the robot model state  
- **joint_state_publisher**：发布关节状态  
  **joint_state_publisher**: Publishes the joint states  
- **ros2_control**：控制插件（例如 `gazebo_ros_control`）  
  **ros2_control**: Control plugin (e.g., `gazebo_ros_control`)  

---

## 注意事项 | Notes  

- 修改机器人结构或参数时，请在 `ur5_parameters.urdf.xacro` 和 `ur5_structure.urdf.xacro` 中同步调整。  
  When modifying robot structure or parameters, update both `ur5_parameters.urdf.xacro` and `ur5_structure.urdf.xacro`.  
- `ur5_main.urdf.xacro` 是最终的实例化文件，勿直接编辑模板文件内容。  
  `ur5_main.urdf.xacro` is the final instantiated file. Do not directly edit template files.  
- 确保 `meshes` 文件夹包含正确的 STL/DAE 文件。  
  Ensure the `meshes` folder contains correct STL/DAE files.  

---

## 贡献 | Contributing  

如果发现问题或需要改进，请提交 Pull Request 或报告 Issue。  
If you find issues or improvements, please submit a Pull Request or report an Issue.  

---

## 维护者 | Maintainer  

- **作者 | Author**: Zezheng Fu (Zell)  
- **联系方式 | Contact**: 11911719@mail.sustech.edu.cn  

--- 

