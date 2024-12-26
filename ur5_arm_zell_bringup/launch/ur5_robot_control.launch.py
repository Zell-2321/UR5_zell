import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
import xacro


def generate_launch_description() -> LaunchDescription:
    # declare arguements 声明参数
    declared_arguments = []

    ## Set perfix of robot model
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='"ur5_1_"',
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )
    ## controllers can be chosen:
        # - forward_position_controller
        # - forward_velocity_controller
        # - forward_acceleration_controller
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_controller",
            default_value="forward_velocity_controller",
            description="Robot controller to start. \
                see: https://control.ros.org/rolling/index.html for more description",
        )
    )
    ## start rViz2
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
    robot_controller = LaunchConfiguration("robot_controller")
    gui = LaunchConfiguration("gui")

    #---------------------------------------------------------------------------------------------------------------------------

    ur5_description_path = os.path.join(
        get_package_share_directory('ur5_arm_zell_description'))

    xacro_file = os.path.join(ur5_description_path,
                              'urdf',
                              'ur5_main.urdf.xacro')

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    robot_description_config = doc.toxml()
    robot_description = {'robot_description': robot_description_config}  

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'robot'],
                        output='screen') 

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", 
                   "--controller-manager", "/controller_manager"],
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["forward_position_controller", "-c", "/controller_manager"],
    )

    # robot_trajectory_controller_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    # )


    return LaunchDescription([
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[robot_controller_spawner],
            )
        ),
        spawn_entity,
        node_robot_state_publisher,
    ])

