import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QHBoxLayout, QSpinBox
from PyQt5.QtCore import Qt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class ArmControllerNode(Node):
    def __init__(self):
        super().__init__('arm_controller_node')
        # 避免与Node内部属性冲突
        self._publishers = {
            'position': self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10),
            'velocity': self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10),
            'effort': self.create_publisher(Float64MultiArray, '/joint_effort_controller/commands', 10)
        }
        self.joint_count = 6

    def publish_command(self, control_type, values):
        msg = Float64MultiArray()
        msg.data = values
        if control_type in self._publishers:
            self._publishers[control_type].publish(msg)
            self.get_logger().info(f'Published {control_type} command: {values}')
        else:
            self.get_logger().error(f'No publisher found for {control_type}')

    def stop_publishers(self):
        # 销毁所有发布器，停止消息发送
        for pub in self._publishers.values():
            pub.destroy()


class ControlGUI(QWidget):
    def __init__(self, arm_controller):
        super().__init__()
        self.arm_controller = arm_controller
        self.setWindowTitle('Robot Arm Control Panel')
        self.setGeometry(300, 300, 600, 400)
        self.layout = QVBoxLayout()
        
        # 自由度设置
        self.layout.addWidget(QLabel("Select Degrees of Freedom (DOF):"))
        self.dof_selector = QSpinBox()
        self.dof_selector.setMinimum(6)
        self.dof_selector.setMaximum(10)
        self.dof_selector.setValue(6)
        self.dof_selector.valueChanged.connect(self.update_dof)
        self.layout.addWidget(self.dof_selector)

        # 滑动条控制器
        self.sliders = []
        self.create_sliders(self.dof_selector.value())

        # 按钮区域
        self.position_button = QPushButton('Send Position Command')
        self.position_button.clicked.connect(lambda: self.send_command('position'))
        self.layout.addWidget(self.position_button)

        self.velocity_button = QPushButton('Send Velocity Command')
        self.velocity_button.clicked.connect(lambda: self.send_command('velocity'))
        self.layout.addWidget(self.velocity_button)

        self.effort_button = QPushButton('Send Effort Command')
        self.effort_button.clicked.connect(lambda: self.send_command('effort'))
        self.layout.addWidget(self.effort_button)

        # 添加复位按键
        self.reset_button = QPushButton('Reset All Commands to 0')
        self.reset_button.clicked.connect(self.reset_commands)
        self.layout.addWidget(self.reset_button)

        # 添加退出按键
        self.exit_button = QPushButton('Exit')
        self.exit_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.exit_button)

        self.setLayout(self.layout)

    def create_sliders(self, dof):
        for slider in self.sliders:
            self.layout.removeWidget(slider)
            slider.deleteLater()
        self.sliders.clear()

        for i in range(dof):
            h_layout = QHBoxLayout()
            label = QLabel(f'Joint {i+1}')
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksBelow)
            
            h_layout.addWidget(label)
            h_layout.addWidget(slider)
            self.sliders.append(slider)
            self.layout.addLayout(h_layout)

    def send_command(self, control_type):
        values = [slider.value() / 10 for slider in self.sliders]
        self.arm_controller.publish_command(control_type, values)

    def reset_commands(self):
        # 将滑块置零
        for slider in self.sliders:
            slider.setValue(0)
        # 发布所有关节置零的指令
        zero_values = [0.0] * len(self.sliders)
        for control_type in self.arm_controller._publishers.keys():
            self.arm_controller.publish_command(control_type, zero_values)

    def update_dof(self):
        new_dof = self.dof_selector.value()
        self.arm_controller.joint_count = new_dof
        self.create_sliders(new_dof)

    def close_application(self):
        # 停止发布器并销毁ROS节点
        self.arm_controller.stop_publishers()
        self.arm_controller.destroy_node()
        rclpy.shutdown()
        QApplication.quit()


def main(args=None):
    rclpy.init(args=args)
    arm_controller = ArmControllerNode()
    
    app = QApplication(sys.argv)
    gui = ControlGUI(arm_controller)
    gui.show()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(arm_controller)
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass
    finally:
        arm_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
