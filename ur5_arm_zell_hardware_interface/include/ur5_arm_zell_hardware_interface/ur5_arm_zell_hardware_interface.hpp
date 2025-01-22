#ifndef ROS2_CONTROL_HARDWARE__ROBOT_SYSTEM_HPP_
#define ROS2_CONTROL_HARDWARE__ROBOT_SYSTEM_HPP_

#include <memory>
#include <string>
#include <vector>

// #include "angles/angles.h"

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

// head file for ros2 subscriber and publisher
#include "rclcpp/rclcpp.hpp"  
#include "std_msgs/msg/float64.hpp"

// TODO: head file for gazebo
// #include "control_toolbox/pid.hpp"
// #include "gazebo_ros2_control/gazebo_system_interface.hpp"

// #include "std_msgs/msg/bool.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace ur5_arm_zell_hardware_interface
{
// TODO: Forward declaration
class GazeboSystemPrivate;

// These class must inherit `gazebo_ros2_control::GazeboSystemInterface` which implements a
// simulated `ros2_control` `hardware_interface::SystemInterface`.

class RobotSystemHardware : public hardware_interface::SystemInterface
{
public:
    RCLCPP_SHARED_PTR_DEFINITIONS(RobotSystemHardware)

    CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override; // 虚函数
    CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override; // 虚函数
    std::vector<hardware_interface::StateInterface> export_state_interfaces() override; // 纯虚函数
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override; // 纯虚函数
    CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override; // 虚函数
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override; // 虚函数
    hardware_interface::return_type read(const rclcpp::Time &time, const rclcpp::Duration &period) override; // 纯虚函数
    hardware_interface::return_type write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) override; // 纯虚函数

private:
    // for gazebo simulation
    // void registerJoints(
    //     const hardware_interface::HardwareInfo & hardware_info,
    //     gazebo::physics::ModelPtr parent_model);
    
    // void registerSensors(
    //     const hardware_interface::HardwareInfo & hardware_info,
    //     gazebo::physics::ModelPtr parent_model);
    
    // bool extractPID(
    //     const std::string & prefix,
    //     const hardware_interface::ComponentInfo & joint_info, control_toolbox::Pid & pid);

    // bool extractPIDFromParameters(
    //     const std::string & control_mode, const std::string & joint_name, control_toolbox::Pid & pid);
    
    // /// \brief Private data class
    // std::unique_ptr<GazeboSystemPrivate> dataPtr;

    // Parameters for the robot hardware interface
    double hw_start_sec_;
    double hw_stop_sec_;
    double hw_slowdown_;

    std::vector<double> hw_position_max_;
    std::vector<double> hw_position_min_;
    std::vector<double> hw_velocity_max_;
    std::vector<double> hw_velocity_min_;
    std::vector<double> hw_acceleration_max_;
    std::vector<double> hw_acceleration_min_;
    std::vector<double> hw_effort_max_;
    std::vector<double> hw_effort_min_;


    // Store the command for the simulated robot
    std::vector<double> hw_commands_position_;
    std::vector<double> hw_commands_velocity_;
    std::vector<double> hw_commands_acceleration_;
    std::vector<double> hw_commands_effort_;
    std::vector<double> hw_states_position_;
    std::vector<double> hw_states_velocity_;
    std::vector<double> hw_states_acceleration_;
    std::vector<double> hw_states_effort_;


    // Add ros2 subscriber to subscriber information from rViz
    // rclcpp::Node::SharedPtr node_;
    // rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr command_subscriber_;

    // void ur5_arm_zell_state_callback(const std_msgs::msg::Float64::SharedPtr msg);
};

}   // namespace ur5_arm_zell_hardware_interface

#endif  // ROS2_CONTROL_HARDWARE__ROBOT_SYSTEM_HPP_