from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
    declared_arguments.append(
        DeclareLaunchArgument(
            "joint_limited",
            default_value="false",
            description="If true, all joint value will between -pi and pi, otherwise, it will be the joint limits in the xacro file.",
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
    ## controllers can be chosen:
        # - forward_position_controller
        # - forward_velocity_controller
        # - forward_acceleration_controller # gazebo hardware interface does not support this controller
        # - joint_effort_controller
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_controller",
            default_value="forward_velocity_controller",
            description="Robot controller to start. \
                see: https://control.ros.org/rolling/index.html for more description",
        )
    )
    
    #---------------------------------------------------------------------------------------------------------------------------

    # Initialize Arguments
    prefix = LaunchConfiguration("prefix")
    joint_limited=LaunchConfiguration("joint_limited")
    robot_controller = LaunchConfiguration("robot_controller")
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

    #---------------------------------------------------------------------------------------------------------------------------
    ## setup file
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("ur5_arm_zell_bringup"),
            "config",
            "controller_configuration.yaml",
        ]
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("ur5_arm_zell_bringup"), "rviz", "rviz_setting.rviz"]
    )

    #---------------------------------------------------------------------------------------------------------------------------
    # Nodes
    # control_node = Node(
    #     package="controller_manager",
    #     executable="ros2_control_node",
    #     parameters=[robot_controllers],
    #     output="both",
    #     remappings=[
    #         ("~/robot_description", "/robot_description"),
    #     ],
    # ) # 如果启动了这个节点，则 description包中ur5_gazebo中不需要再次声明插件，否则会出现
    # [ERROR] [1716894885.022794553] [controller_manager]: The published robot description file (urdf) seems not to be genuine. 
    # The following error was caught:According to the loaded plugin descriptions the class gazebo_ros2_control/GazeboSystem with base 
    # class type hardware_interface::SystemInterface does not exist. Declared types are  fake_components/GenericSystem mock_components/GenericSystem
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(gui),
    )
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'robot'],
                        output='screen')

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[robot_controller, "--controller-manager", "/controller_manager"],
    )

    #---------------------------------------------------------------------------------------------------------------------------
    # Delay rviz start after `joint_state_broadcaster`
    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        )
    )

    # Delay start of joint_state_broadcaster after `robot_controller`
    # TODO(anyone): This is a workaround for flaky tests. Remove when fixed.
    delay_joint_state_broadcaster_after_robot_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_controller_spawner,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    nodes = [
        spawn_entity,
        # control_node,
        robot_state_pub_node,
        robot_controller_spawner,
        delay_rviz_after_joint_state_broadcaster_spawner,
        delay_joint_state_broadcaster_after_robot_controller_spawner,
    ]

    return LaunchDescription(declared_arguments + nodes)

