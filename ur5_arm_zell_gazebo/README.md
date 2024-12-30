# UR5 Arm Gazebo Simulation Package  
# UR5机械臂 Gazebo 仿真包  

## 简介 | Overview  
该包用于在 Gazebo 中仿真 UR5 机械臂，包含启动文件以及 Gazebo 世界配置。  
This package is for simulating the UR5 robotic arm in Gazebo, including launch files and Gazebo world configurations.  

---

## 文件夹结构 | Directory Structure  

```plaintext
.
├── include           # 头文件目录  
│                   # Header files directory  
├── launch            # ROS 2 启动文件，负责仿真和模型加载  
│   ├── spawn_robot_description.launch.py  
│   ├── spawn_robot_ros2.launch.xml  
│   ├── start_world.launch.py  
│   └── urdf_visualize.launch.py  
│                   # ROS 2 launch files for spawning robot and world simulation  
├── models            # Gazebo 3D模型目录  
│                   # Gazebo 3D models directory  
├── src               # C++/Python 源代码目录  
│                   # Source code directory  
├── worlds            # Gazebo 世界文件，定义仿真环境  
│                   # Gazebo world files defining simulation environments  
├── CMakeLists.txt    # 构建系统配置文件  
│                   # Build system configuration file  
├── launch.sh         # 启动脚本文件，快速启动仿真环境  TODO
│                   # Shell script to quickly launch simulations  
├── LICENSE           # 许可证文件  
│                   # License file  
├── package.xml       # ROS 2 包配置文件  
│                   # ROS 2 package configuration file  
└── README.md         # 当前文档  
│                   # This document  
```

---

## 主要文件说明 | Key Files Description  

1. **`spawn_robot_description.launch.py`**  
   - 功能：加载 UR5 机械臂描述文件，并在 Gazebo 中生成模型。  
     Function: Loads UR5 arm description and spawns the model in Gazebo.  

2. **`spawn_robot_ros2.launch.xml`**  
   - 功能：ROS 2 启动文件，使用 XML 格式加载 UR5 机械臂描述。  
     Function: ROS 2 launch file in XML format to load UR5 robot description.  

3. **`start_world.launch.py`**  
   - 功能：启动 Gazebo 世界。  
     Function: Starts the Gazebo world.  

4. **`urdf_visualize.launch.py`**  
   - 功能：在 RViz2 中可视化 URDF 模型，验证模型完整性。  
     Function: Visualizes the URDF model in RViz2 to verify completeness.  

5. **`worlds` 目录**  
   - 存储 Gazebo 世界文件，定义仿真环境，包括障碍物和地形。  
     Contains Gazebo world files defining simulation environments, obstacles, and terrain.  

---

## 如何使用 | How to Use  

1. **启动 Gazebo 世界 | Start Gazebo**  
   使用以下命令启动 Gazebo 并生成 UR5 模型：  
   ```bash
   ros2 launch ur5_arm_zell_gazebo start_world.launch.py
   ```

2. **在 RViz2 以及 Gazebo中可视化 URDF 模型 | Visualize URDF Model in RViz2 and Gazebo**  
   ```bash
      ros2 launch ur5_arm_zell_gazebo spawn_robot_ros2.launch.xml
   ```
---

## 依赖项 | Dependencies  

- **ROS 2**（推荐使用 Humble版本）  
  **ROS 2** (Recommended Humble or above)  
- **Gazebo**（9 或以上版本）  
  **Gazebo** (9 or later)  
- **gazebo_ros**  
- **joint_state_publisher**  
- **xacro**  
- **ros2_control**  
- **ur5_arm_zell_description**

---

## 注意事项 | Notes  


---

## 贡献 | Contributing  

如果发现问题或改进建议，请提交 Pull Request 或报告 Issue。  
If you find issues or improvements, please submit a Pull Request or report an Issue.  

---

## 维护者 | Maintainer  

- **作者 | Author**: Zezheng Fu (Zell)  
- **联系方式 | Contact**: 11911719@mail.sustech.edu.cn  
```  
