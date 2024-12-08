from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

"""
Launch file to load yaml config to the ros2 parameters service
"""
def generate_launch_description() -> LaunchDescription:
    # Path to YAML file
    robot_params_file = os.path.join(
        get_package_share_directory('ur5_arm_zell_description'),
        'config',
        'robot_sensor.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_params_file', default_value=robot_params_file),

        # Load parameters from YAML file
        Node(
            package='rclpy',
            executable='parameter_bridge',
            name='parameter_loader',
            namespace='ur5_zell_robot',
            parameters=[LaunchConfiguration('robot_params_file')],
            output='screen'
        )
    ])
