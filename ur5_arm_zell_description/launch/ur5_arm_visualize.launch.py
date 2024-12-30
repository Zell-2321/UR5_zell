import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

"""
Launch file to load a specified xacro file
modified the "xacro_file" parameters if you want to change xacro file
"""
def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value="ur5_1_",
            description="Prefix of the joint names, useful for multi-robot setup. ",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "joint_limited",
            default_value="false",
            description="If true, all joint value will between -pi and pi, otherwise, it will be the joint limits in the xacro file.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Start RViz2 automatically with this launch file.",
        )
    )

    #---------------------------------------------------------------------------------------------------------------------------

    # Initialize Arguments
    prefix = LaunchConfiguration("prefix")
    joint_limited = LaunchConfiguration("joint_limited")
    gui = LaunchConfiguration("gui")

    #---------------------------------------------------------------------------------------------------------------------------
    # convert XACRO file into URDF
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("ur5_arm_zell_description"),
                    "urdf",
                    "ur5_main.urdf.xacro",
                ]
            ),
            " ",
            "prefix:=",
            prefix,
            " ",
            "joint_limited:=",
            joint_limited,
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # ros2 nodes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    rviz_config_dir = os.path.join(get_package_share_directory("ur5_arm_zell_description"), 'rviz', 'ur5_arm_zell_visualize.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_dir],
        condition=IfCondition(gui),
    )

    nodes = [
        robot_state_publisher,
        joint_state_publisher_gui_node,
        rviz_node,
    ]

    return LaunchDescription(declared_arguments + nodes)