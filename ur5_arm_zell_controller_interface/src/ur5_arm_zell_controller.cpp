#include "ur5_arm_zell_controller_interface/ur5_arm_zell_controller.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ur5_arm_zell_controller
{
RobotController::RobotController() : controller_interface::ControllerInterface() {}

controller_interface::CallbackReturn RobotController::on_init() 
{
    try {
        auto_declare("joints", std::vector<std::string>());
        auto_declare("interface_name", std::string());
    } catch (const std::exception &e) {
        fprintf(stderr, "Exception thrown during init stage with message: %s \n",
                e.what());
        return CallbackReturn::ERROR;
    }

    return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration RobotController::command_interface_configuration() const
{
    controller_interface::InterfaceConfiguration command_interfaces_config;
    command_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

    command_interfaces_config.names.reserve(joint_names_.size());
    for (const auto & joint : joint_names_) {
        command_interfaces_config.names.push_back(joint + "/" + interface_name_);
    }

    return command_interfaces_config;
}

controller_interface::InterfaceConfiguration RobotController::state_interface_configuration() const
{
    controller_interface::InterfaceConfiguration state_interfaces_config;
    state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

    state_interfaces_config.names.reserve(joint_names_.size());
    for (const auto & joint : joint_names_) {
        state_interfaces_config.names.push_back(joint + "/" + interface_name_);
    }

    return state_interfaces_config;
}

template <typename T>
bool get_ordered_interfaces(
  std::vector<T> & unordered_interfaces, const std::vector<std::string> & joint_names,
  const std::string & interface_type, std::vector<std::reference_wrapper<T>> & ordered_interfaces)
{
    for (const auto & joint_name : joint_names) {
        for (auto & command_interface : unordered_interfaces) {
            if (
                (command_interface.get_name() == joint_name) &&
                (command_interface.get_interface_name() == interface_type)) {
                ordered_interfaces.push_back(std::ref(command_interface));
            }
        }
    }

  return joint_names.size() == ordered_interfaces.size();
}

controller_interface::CallbackReturn RobotController::on_configure(const rclcpp_lifecycle::State & /*previous_state*/) 
{
    auto error_if_empty = [&](const auto &parameter, const char *parameter_name) {
        if (parameter.empty()) {
        RCLCPP_ERROR(get_node()->get_logger(), "'%s' parameter was empty",
                    parameter_name);
        return true;
        }
        return false;
    };

    auto get_string_array_param_and_error_if_empty =
        [&](std::vector<std::string> &parameter, const char *parameter_name) {
            parameter = get_node()->get_parameter(parameter_name).as_string_array();
            return error_if_empty(parameter, parameter_name);
        };

    auto get_string_param_and_error_if_empty =
    [&](std::string &parameter, const char *parameter_name) {
        parameter = get_node()->get_parameter(parameter_name).as_string();
        return error_if_empty(parameter, parameter_name);
    };

    if (
        get_string_array_param_and_error_if_empty(joint_names_, "joints") ||
        get_string_param_and_error_if_empty(interface_name_, "interface_name")) {
        return CallbackReturn::ERROR;
    }

    // Command Subscriber and callbacks
    auto callback_command =
        [&](const std::shared_ptr<ControllerCommandMsg> msg) -> void {
        if (msg->joint_names.size() == joint_names_.size()) {
        input_command_.writeFromNonRT(msg);
        } else {
        RCLCPP_ERROR(get_node()->get_logger(),
                    "Received %zu , but expected %zu joints in command. "
                    "Ignoring message.",
                    msg->joint_names.size(), joint_names_.size());
        }
    };
    command_subscriber_ = get_node()->create_subscription<ControllerCommandMsg>(
        "~/commands", rclcpp::SystemDefaultsQoS(), callback_command);

    // State publisher
    s_publisher_ =
    get_node()->create_publisher<ControllerStateMsg>(
        "~/state", rclcpp::SystemDefaultsQoS());
    state_publisher_ = std::make_unique<ControllerStatePublisher>(s_publisher_);

    state_publisher_->lock();
    state_publisher_->msg_.header.frame_id = joint_names_[0];
    state_publisher_->unlock();

    RCLCPP_INFO_STREAM(get_node()->get_logger(), "configure successful");
    return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn RobotController::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    // Set default value in command
    std::shared_ptr<ControllerCommandMsg> msg = std::make_shared<ControllerCommandMsg>();
    msg->joint_names = joint_names_;
    msg->displacements.resize(joint_names_.size(), std::numeric_limits<double>::quiet_NaN());
    input_command_.writeFromNonRT(msg);

    return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn RobotController::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    return CallbackReturn::SUCCESS;
}

controller_interface::return_type RobotController::update(const rclcpp::Time &time, const rclcpp::Duration & /*period*/) 
{
    auto current_command = input_command_.readFromRT();

    for (size_t i = 0; i < command_interfaces_.size(); ++i) {
        if (!std::isnan((*current_command)->displacements[i])) {
            command_interfaces_[i].set_value((*current_command)->displacements[i]);
        }
    }

    if (state_publisher_ && state_publisher_->trylock()) {
        state_publisher_->msg_.header.stamp = time;
        state_publisher_->msg_.set_point = command_interfaces_[0].get_value();

        state_publisher_->unlockAndPublish();
    }

    return controller_interface::return_type::OK;
}

controller_interface::CallbackReturn RobotController::on_cleanup(const rclcpp_lifecycle::State &)
{
    return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn RobotController::on_error(const rclcpp_lifecycle::State &)
{
    return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn RobotController::on_shutdown(const rclcpp_lifecycle::State &)
{
    return CallbackReturn::SUCCESS;
}

}  // namespace ur5_arm_zell_controller

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(ur5_arm_zell_controller::RobotController, controller_interface::ControllerInterface)