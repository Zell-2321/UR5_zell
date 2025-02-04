import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
from datetime import datetime


class JointStateListener(Node):
    def __init__(self):
        super().__init__('joint_state_listener')
        
        # 初始化订阅器，监听 `/joint_states` 话题
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10  # 队列大小
        )
        
        # 初始化 CSV 文件
        self.file_name = 'joint_states_data.csv'
        self.file = open(self.file_name, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)

        # 写入 CSV 文件的表头
        self.csv_writer.writerow(['timestamp', 'time', 'name', 'position', 'velocity'])

        self.get_logger().info('Joint state listener started!')

    def joint_state_callback(self, msg):
        # 获取当前时间戳
        timestamp_sec = msg.header.stamp.sec
        timestamp_nanosec = msg.header.stamp.nanosec

        # 遍历每个关节的状态数据并写入到 CSV 文件中
        # for i in range(len(msg.name)):
        joint_name = msg.name[1]
        joint_position = msg.position[1]
        joint_velocity = msg.velocity[1]
        # joint_effort = msg.effort[i] if len(msg.effort) > 0 else None  # 一些情况可能没有effort数据

        # 将数据写入 CSV
        self.csv_writer.writerow([timestamp_sec, timestamp_nanosec, joint_name, joint_position, joint_velocity])

    def __del__(self):
        # 关闭文件
        self.file.close()


def main(args=None):
    rclpy.init(args=args)

    # 创建节点实例
    node = JointStateListener()

    # 保持节点运行
    rclpy.spin(node)

    # 关闭节点
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
