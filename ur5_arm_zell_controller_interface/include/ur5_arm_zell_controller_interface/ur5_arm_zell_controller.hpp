#ifndef ROBOT_CONTROLLER__ROBOT_CONTROLLER_HPP_
#define ROBOT_CONTROLLER__ROBOT_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"

#include "control_msgs/msg/joint_controller_state.hpp"
#include "control_msgs/msg/joint_jog.hpp"

namespace ur5_arm_zell_controller 
{
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class RobotController : public controller_interface::ControllerInterface 
{
public:
    RobotController();

    controller_interface::InterfaceConfiguration command_interface_configuration() const override;
    controller_interface::InterfaceConfiguration state_interface_configuration() const override;
    controller_interface::return_type update(const rclcpp::Time &time, const rclcpp::Duration &period) override;
    controller_interface::CallbackReturn on_init() override;
    controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &previous_state) override;
    controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;
    controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;
    controller_interface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State &previous_state) override;
    controller_interface::CallbackReturn on_error(const rclcpp_lifecycle::State &previous_state) override;
    controller_interface::CallbackReturn on_shutdown(const rclcpp_lifecycle::State &previous_state) override;

protected:
    std::vector<std::string> joint_names_;
    std::string interface_name_;
    std::vector<std::string> command_interface_types_;
    std::vector<std::string> state_interface_types_;

    // Command subscribers and Controller State publisher
    using ControllerCommandMsg = control_msgs::msg::JointJog;

    rclcpp::Subscription<ControllerCommandMsg>::SharedPtr command_subscriber_ = nullptr;
    realtime_tools::RealtimeBuffer<std::shared_ptr<ControllerCommandMsg>> input_command_;

    using ControllerStateMsg = control_msgs::msg::JointControllerState;
    using ControllerStatePublisher = realtime_tools::RealtimePublisher<ControllerStateMsg>;

    rclcpp::Publisher<ControllerStateMsg>::SharedPtr s_publisher_;
    std::unique_ptr<ControllerStatePublisher> state_publisher_;
};

} // namespace ur5_arm_zell_controller

#endif // ROBOT_CONTROLLER__ROBOT_CONTROLLER_HPP_