#ifndef ROS2_CONTROL_DEMO_HARDWARE__RRBOT_SYSTEM_HPP_
#define ROS2_CONTROL_DEMO_HARDWARE__RRBOT_SYSTEM_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace ur5_arm_zell_hardware_interface
{
class RobotSystemHardware : public hardware_interface::SystemInterface
{
public:
    RCLCPP_SHARED_PTR_DEFINITIONS(RobotSystemHardware);

    CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override; // 虚函数

    CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override; // 虚函数

    std::vector<hardware_interface::StateInterface> export_state_interfaces() override; // 纯虚函数

    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override; // 纯虚函数

    CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override; // 虚函数

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override; // 虚函数

    hardware_interface::return_type read(const rclcpp::Time &time, const rclcpp::Duration &period) override; // 纯虚函数

    hardware_interface::return_type write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) override; // 纯虚函数

private:
    // Parameters for the robot simulation
    double hw_start_sec_;
    double hw_stop_sec_;
    double hw_slowdown_;

    // Store the command for the simulated robot
    std::vector<double> hw_commands_;
    std::vector<double> hw_states_;
};

}   // namespace ur5_arm_zell_hardware_interface

#endif 