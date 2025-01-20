#include "ur5_arm_zell_hardware_interface/ur5_arm_zell_hardware_interface.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace ur5_arm_zell_hardware_interface
{
CallbackReturn RobotSystemHardware::on_init(const hardware_interface::HardwareInfo & info)
{
    // setup communication with robot hardware
    // ...
    static const std::unordered_set<std::string> valid_command_interfaces = {
    hardware_interface::HW_IF_POSITION,
    hardware_interface::HW_IF_VELOCITY,
    hardware_interface::HW_IF_ACCELERATION,
    hardware_interface::HW_IF_EFFORT
    };
    static const std::unordered_set<std::string> valid_state_interfaces = {
    hardware_interface::HW_IF_POSITION,
    hardware_interface::HW_IF_VELOCITY,
    hardware_interface::HW_IF_ACCELERATION,
    hardware_interface::HW_IF_EFFORT
    };
    if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) // info -> hardware_interface::SystemInterface->info_
    {
        return CallbackReturn::ERROR;
    }
    // TODO: Remove after finish
    // START: This part here is for exemplary purposes - Please do not copy to your production code
    hw_start_sec_ = stod(info_.hardware_parameters["example_param_hw_start_duration_sec"]);
    hw_stop_sec_ = stod(info_.hardware_parameters["example_param_hw_stop_duration_sec"]);
    hw_slowdown_ = stod(info_.hardware_parameters["example_param_hw_slowdown"]);
    // END: This part here is for exemplary purposes - Please do not copy to your production code

    // TODO: add limit according to urdf

    hw_commands_position_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_commands_velocity_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_commands_acceleration_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_commands_effort_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_states_position_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_states_velocity_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_states_acceleration_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_states_effort_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_position_max_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_position_min_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_velocity_max_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_velocity_min_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_acceleration_max_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_acceleration_min_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_effort_max_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_effort_min_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());

    for (const hardware_interface::ComponentInfo & joint : info_.joints)
    {
        // 打印command_interfaces 以及 state_interfaces 的大小
        RCLCPP_INFO(
            rclcpp::get_logger("RobotSystemHardware"),
            "%zu command interfaces found, %zu state interfaces found.",
            joint.command_interfaces.size(),
            joint.state_interfaces.size());
        
        // 确保command interface在可选的interface之间
        for (const auto & command_interface : joint.command_interfaces)
        {
            if (valid_command_interfaces.find(command_interface.name) == valid_command_interfaces.end())
            {
                RCLCPP_FATAL(
                    rclcpp::get_logger("RobotSystemHardware"),
                    "Joint '%s' has an unsupported command interface: '%s'.",
                    joint.name.c_str(), command_interface.name.c_str());
                return CallbackReturn::ERROR;
            }
        }

        for (const auto & state_interface : joint.state_interfaces)
        {
            if (valid_state_interfaces.find(state_interface.name) == valid_state_interfaces.end())
            {
                RCLCPP_FATAL(
                    rclcpp::get_logger("RobotSystemHardware"),
                    "Joint '%s' has an unsupported state interface: '%s'.",
                    joint.name.c_str(), state_interface.name.c_str());
                return CallbackReturn::ERROR;
            }
        }
        

        // 在此添加其他硬件相关初始化步骤 (Add other hardware initialization steps here)
    }
    
    for (size_t i = 0; i < info.joints.size(); ++i) {
        const auto &joint = info.joints[i];

        // 遍历 command_interfaces
        for (const auto &command_interface : joint.command_interfaces) {
            const std::string &interface_name = command_interface.name;

            // 解析 min 和 max 参数
            double min_value = std::stod(command_interface.min);
            double max_value = std::stod(command_interface.max);

            if (interface_name == "position") {
                hw_position_min_[i] = min_value;
                hw_position_max_[i] = max_value;
            } else if (interface_name == "velocity") {
                hw_velocity_min_[i] = min_value;
                hw_velocity_max_[i] = max_value;
            } else if (interface_name == "acceleration") {
                hw_acceleration_min_[i] = min_value;
                hw_acceleration_max_[i] = max_value;
            } else if (interface_name == "effort") {
                hw_effort_min_[i] = min_value;
                hw_effort_max_[i] = max_value;
            } else {
                RCLCPP_WARN(rclcpp::get_logger("RobotSystemHardware"),
                            "Unknown command interface: %s", interface_name.c_str());
            }
        }
    }

    return CallbackReturn::SUCCESS;
}

CallbackReturn RobotSystemHardware::on_configure(const rclcpp_lifecycle::State &previous_state) 
{
    // Unconfigured -> Inactive

    // 防止报错 prevent unused variable warning
    auto prev_state = previous_state;
    RCLCPP_INFO(rclcpp::get_logger("RobotSystemHardware"),
                "Configuring ...please wait...");

    // 模拟硬件启动时间 simulate hardware initialization time.
    for (int i = 0; i < hw_start_sec_; i++) {
        rclcpp::sleep_for(std::chrono::seconds(1));
        RCLCPP_INFO(rclcpp::get_logger("RobotSystemHardware"),
                    "%.1f seconds left...", hw_start_sec_ - i);
    }

    // reset values always when configuring hardware
    for (uint i = 0; i < hw_states_position_.size(); i++) {
        hw_commands_position_[i] = 0;
        hw_commands_velocity_[i] = 0;
        hw_commands_acceleration_[i] = 0;
        hw_commands_effort_[i] = 0;

        hw_states_position_[i] = 0;
        hw_states_velocity_[i] = 0;
        hw_states_acceleration_[i] = 0;
        hw_states_effort_[i] = 0;
    }
    // 读取、解析参数（如硬件参数、控制器参数等）
    // 分配或初始化资源（如内存、数据结构）
    // 打开与硬件（或模拟器）的基本通信通道
    // 设置一些必要的初始条件，但尚未开始真正的控制或发布命令
    // Read and parse parameters (e.g., hardware parameters, controller parameters, etc.)
    // Allocate or initialize resources (e.g., memory, data structures)
    // Open basic communication channels with the hardware (or simulator)
    // Set up necessary initial conditions, but do not start actual control or publish commands yet


    RCLCPP_INFO(rclcpp::get_logger("RobotSystemHardware"),
                "Successfully configured!");

    return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> RobotSystemHardware::export_state_interfaces()
{
    // add state interfaces to ``state_interfaces`` for each joint, e.g. `info_.joints[0].state_interfaces_`, `info_.joints[1].state_interfaces_`, `info_.joints[2].state_interfaces_` ...
    // ...
    std::vector<hardware_interface::StateInterface> state_interfaces;
    for (uint i = 0; i < info_.joints.size(); i++)
    {
        // 位置信息 position information
        state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_states_position_[i]));

        // 速度信息 veloity information
        state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_states_velocity_[i])); 

        // 加速度信息 acceleration information
        state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_ACCELERATION, &hw_states_acceleration_[i])); 

        // 力矩信息 effort information
        state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_states_effort_[i])); 

    }
    return state_interfaces;
}
    
std::vector<hardware_interface::CommandInterface> RobotSystemHardware::export_command_interfaces()
{
    // add command interfaces to ``command_interfaces`` for each joint, e.g. `info_.joints[0].command_interfaces_`, `info_.joints[1].command_interfaces_`, `info_.joints[2].command_interfaces_` ...
    // ...
    std::vector<hardware_interface::CommandInterface> command_interfaces;
    for (uint i = 0; i < info_.joints.size(); i++)
    {
        command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_position_[i]));

        command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_commands_velocity_[i]));

        command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_ACCELERATION, &hw_commands_acceleration_[i]));

        command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_commands_effort_[i]));
    }

    return command_interfaces;
}

CallbackReturn RobotSystemHardware::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    // Inactive -> Active
    // 准备硬件设备、启动实时线程或重置命令/状态、延时操作 Prepare hardware devices, start real-time threads, reset commands/states, and perform delay operations.
    RCLCPP_INFO(
        rclcpp::get_logger("RobotSystemHardware"), "Activating ...please wait...");

    // 延时操作，模拟硬件初始化时间 Delayed operations to simulate hardware initialization time.
    for (int i = 0; i < hw_start_sec_; i++)
    {
        rclcpp::sleep_for(std::chrono::seconds(1));
        RCLCPP_INFO(
        rclcpp::get_logger("RobotSystemHardware"), "%.1f seconds left...",
        hw_start_sec_ - i);
        // activate your hardware here
    }

    // command and state should be equal when starting
    for (uint i = 0; i < hw_states_position_.size(); i++)
    {
        hw_commands_position_[i] = hw_states_position_[i];
        hw_commands_velocity_[i] = hw_states_velocity_[i];
        hw_commands_acceleration_[i] = hw_states_acceleration_[i];
        hw_commands_effort_[i] = hw_states_effort_[i];
    }
    // 真正启动硬件（或仿真）的运行
    // （如果需要）启动实时线程
    // 重置命令与状态、让硬件或控制器准备好执行控制
    // 可以做延时操作，模拟硬件初始化过程所需时间
    // Actually start hardware (or simulation) operation
    // (If needed) start real-time threads
    // Reset commands and states, ensuring the hardware or controller is ready to execute control
    // Perform delay operations if necessary, simulating the time required for hardware initialization


    RCLCPP_INFO(rclcpp::get_logger("RobotSystemHardware"), "Successfully activated!");

    return CallbackReturn::SUCCESS;
}

CallbackReturn RobotSystemHardware::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    // // Active -> Inactive
    RCLCPP_INFO(
        rclcpp::get_logger("RobotSystemHardware"), "Deactivating ...please wait...");

    for (int i = 0; i < hw_stop_sec_; i++)
    {
        rclcpp::sleep_for(std::chrono::seconds(1));
        RCLCPP_INFO(
            rclcpp::get_logger("RobotSystemHardware"), "%.1f seconds left...",
            hw_stop_sec_ - i);
    }

    RCLCPP_INFO(rclcpp::get_logger("RobotSystemHardware"), "Successfully deactivated!");

    // 停止或关闭硬件控制信号：将输出命令置零、关闭相关通信接口等。
    // 释放或清理资源：关闭不再需要的文件句柄、数据结构、实时线程等。
    // 让硬件进入安全或空闲状态：避免在“非激活”状态时仍驱动执行器、发出力矩或运动指令。
    // 必要的日志与延时：若硬件停机需要一定时间，可以在此处进行延时或输出日志，提示用户当前正在停机。
    // Stop or disable hardware control signals: set output commands to zero, close relevant communication interfaces, etc.
    // Release or clean up resources: close unneeded file handles, data structures, real-time threads, etc.
    // Put the hardware into a safe or idle state: prevent actuators from being driven or torque/commands from being sent when in "inactive" state.
    // Log and delay if necessary: if hardware shutdown takes time, perform the delay here and log information to inform users about the ongoing shutdown.


    return CallbackReturn::SUCCESS;
}

hardware_interface::return_type RobotSystemHardware::read(const rclcpp::Time &time, const rclcpp::Duration &period)
{
    // read hardware values for state interfaces, e.g joint encoders and sensor readings
    // ...
    RCLCPP_INFO(rclcpp::get_logger("RobotSystemHardware"), "Reading...");
    RCLCPP_INFO(
        rclcpp::get_logger("RobotSystemHardware"),
        "Period: %.9f seconds, %ld nanoseconds. time:%.9f seconds, %ld nanoseconds",
        period.seconds(),
        period.nanoseconds(),
        time.seconds(),
        time.nanoseconds());

    for (uint i = 0; i < hw_states_position_.size(); i++)
    {
        // Simulate robot movement
        // effort and acceleration can be controled immediately with noise

        // hw_states_effort_[i] = std::clamp(hw_commands_effort_[i], hw_effort_min_[i], hw_effort_max_[i]); // + Noise
        // hw_states_acceleration_[i] = std::clamp(hw_commands_acceleration_[i], hw_acceleration_min_[i], hw_acceleration_max_[i]); // + Noise

        // hw_states_velocity_[i] = std::clamp(hw_commands_velocity_[i], hw_velocity_min_[i], hw_velocity_max_[i]);
        // hw_states_position_[i] = std::clamp(hw_commands_position_[i], hw_position_min_[i], hw_position_max_[i]);

        // Position control only
        hw_states_position_[i] = hw_states_position_[i] + (hw_commands_position_[i] - hw_states_position_[i]) / hw_slowdown_;
        RCLCPP_INFO(
            rclcpp::get_logger("RobotSystemHardware"), "Got state %.5f for joint %d!",
            hw_states_position_[i], i);
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotSystemHardware"), "Joints successfully read!");


    return hardware_interface::return_type::OK;
}

hardware_interface::return_type RobotSystemHardware::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // START: This part here is for exemplary purposes - Please do not copy to your production code
  RCLCPP_INFO(rclcpp::get_logger("RobotSystemHardware"), "Writing...");

//   for (uint i = 0; i < hw_commands_position_.size(); i++)
//   {
//     // Simulate sending commands to the hardware
//     RCLCPP_INFO(
//       rclcpp::get_logger("RobotSystemHardware"), "Got command %.5f for joint %d!",
//       hw_commands_position_[i], i);
//   }
//   RCLCPP_INFO(
//     rclcpp::get_logger("RobotSystemHardware"), "Joints successfully written!");
  // END: This part here is for exemplary purposes - Please do not copy to your production code

  return hardware_interface::return_type::OK;
}

}   // namespace ur5_arm_zell_hardware_interface

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(ur5_arm_zell_hardware_interface::RobotSystemHardware, hardware_interface::SystemInterface)