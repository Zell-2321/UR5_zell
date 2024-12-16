## Xacro file 文件说明

- ur5_main.urdf.xacro: 实例化了机械臂，包含了ros2_control以及相应的gazebo插件
- ur5_gazebo.urdf.xacro: (模板文件) gazebo 相关插件的定义
- ur5_parameters.urdf.xacro: (模板文件) 所有xacro文件的参数定义, 仅在实例化文件中加载，优先加载
- ur5_structure.urdf.xacro: (模板文件) 机械臂的结构定义，包含了传感器
- ur5_control.urdf.xacro: (模板文件)UR5 机械臂相关控制器定义
