
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# spawn_robot_description.launch.py
import random
from launch_ros.actions import Node
from launch import LaunchDescription

# this is the function launch  system will look for

def generate_launch_description():
    # Position and orientation
    # [X, Y, Z]
    position = [0.0, 0.0, 0.0]
    # [Roll, Pitch, Yaw]
    orientation = [0.0, 0.0, 0.0]
    # Base Name or robot
    robot_base_name = "ur5"

    entity_name = robot_base_name+"-"+str(int(random.random()*100000))

    # Spawn ROBOT Set Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_entity',
        output='screen',
        arguments=['-entity',
                   entity_name,
                   '-x', str(position[0]), '-y', str(position[1]
                                                     ), '-z', str(position[2]),
                   '-R', str(orientation[0]), '-P', str(orientation[1]
                                                        ), '-Y', str(orientation[2]),
                   '-topic', '/robot_description'
                   ]
    )


    # create and return launch description object
    return LaunchDescription(
        [
            spawn_robot,
        ]
    )

# GAZEBO MODELS PATH==/usr/share/gazebo/../../share/gazebo-11/models:/usr/share/gazebo-11/models:/home/zell/ros2_ws/install/my_box_bot_gazebo/share/my_box_bot_gazebo/models:/home/zell/ros2_ws/install/ur5_arm_zell_description/share:/home/zell/ros2_ws/install/ur5_arm_zell_gazebo/share/ur5_arm_zell_gazebo/models
# GAZEBO PLUGINS PATH==/usr/share/gazebo/../../lib/x86_64-linux-gnu/gazebo-11/plugins::/home/zell/ros2_ws/install/ur5_arm_zell_description/lib

# export GAZEBO_PLUGIN_PATH=/usr/share/gazebo/../../share/gazebo-11/models:/usr/share/gazebo-11/models:$GAZEBO_PLUGIN_PATH
# export GAZEBO_MODEL_PATH=/usr/share/gazebo/../../lib/x86_64-linux-gnu/gazebo-11/plugins:$GAZEBO_MODEL_PATH
