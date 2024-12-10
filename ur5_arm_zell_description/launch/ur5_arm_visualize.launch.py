import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

"""
Launch file to load a specified xacro file
modified the "xacro_file" parameters if you want to change xacro file
"""
def generate_launch_description() -> LaunchDescription:
    robot_model_path = os.path.join(
        get_package_share_directory('ur5_arm_zell_description'))

    xacro_file = os.path.join(robot_model_path, 'urdf', 'ur5_main.urdf.xacro')

    # convert XACRO file into URDF
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml(), }

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher_gui_node,
        rviz_node
    ])