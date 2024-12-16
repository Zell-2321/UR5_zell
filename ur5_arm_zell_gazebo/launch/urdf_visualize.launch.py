import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

# this is the function launch  system will look for
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

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        name='joint_state_publisher'
    )

    # RVIZ Configuration
    # rviz_config_dir = os.path.join(get_package_share_directory('ur5_arm_zell_description'), 'rviz', 'urdf_vis.rviz')


    rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            name='rviz_node',
            parameters=[{'use_sim_time': True}],
            # arguments=['-d', rviz_config_dir]
            )

    # create and return launch description object
    return LaunchDescription(
        [            
            robot_state_publisher,
            rviz_node,
            joint_state_publisher,
        ]
    )