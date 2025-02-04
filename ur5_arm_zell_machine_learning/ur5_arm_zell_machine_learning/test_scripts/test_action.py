import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# or_trajectory_data = {
#     1: {'x': 0.0, 'v': 0.0, 'a': 0.0},
#     2: {'x': 1/24, 'v': 1/8, 'a': 1/4},
#     3: {'x': 7/24, 'v': 3/8, 'a': 1/4},
#     4: {'x': 3/4, 'v': 1/2, 'a': 0.0},
#     5: {'x': 29/24, 'v': 3/8, 'a': -1/4},
#     6: {'x': 35/24, 'v': 1/8, 'a': -1/4},
#     7: {'x': 3/2, 'v': 0.0, 'a': 0.0},
#     # 8: {'x': 0.0, 'v': 0.0, 'a': 0.0}
# }
or_trajectory_data = {
    1: {'x': 0.0, 'v': 0.0, 'a': 0.0},
    2: {'x': 0.3, 'v': 1/8, 'a': 1/4},
    3: {'x': 0.6, 'v': 3/8, 'a': 1/4},
    4: {'x': 0.0, 'v': 1/2, 'a': 0.0},
    5: {'x': 0.2, 'v': 3/8, 'a': -1/4},
    6: {'x': 0.2, 'v': 1/8, 'a': -1/4},
    7: {'x': 0.0, 'v': 0.0, 'a': 0.0},
    # 8: {'x': 0.0, 'v': 0.0, 'a': 0.0}
}

# 创建一个新的字典存储修改后的数据
trajectory_data = {}

# 将数据除以4并存储到新的字典中
for t, data in or_trajectory_data.items():
    trajectory_data[t] = {
        'x': data['x'] / 4,
        'v': data['v'] / 4,
        'a': data['a'] / 4
    }

def create_trajectory_point(t, data):
    return JointTrajectoryPoint(
        positions=[data['x'], data['x'], data['x'], data['x'], data['x'], data['x']],  # 所有关节位置相同
        # velocities=[data['v'], data['v'], data['v'], data['v'], data['v'], data['v']],  # 所有关节速度相同
        # accelerations=[data['a'], data['a'], data['a'], data['a'], data['a'], data['a']],  # 所有关节加速度相同
        time_from_start=Duration(sec=t)  # 时间从0开始，每次增加1秒
    )

class TrajectoryClient:
    def __init__(self, node):
        self.node = node
        self.client = ActionClient(
            self.node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_goal(self, joint_names, points):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points = points

        self.node.get_logger().info('Waiting for action server...')
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error('Action server not available!')
            return None

        self.node.get_logger().info('Sending trajectory goal...')
        return self.client.send_goal_async(goal_msg)


class MyNode(Node):
    def __init__(self):
        super().__init__('trajectory_client_node')
        self.get_logger().info('Trajectory client node has started!')


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    # 定义关节名称
    joint_names = [
        'ur5_1_shoulder_pan_joint', 'ur5_1_shoulder_lift_joint',
        'ur5_1_elbow_joint', 'ur5_1_wrist_1_joint',
        'ur5_1_wrist_2_joint', 'ur5_1_wrist_3_joint'
    ]

    # 定义轨迹点
    # points = [
    #     JointTrajectoryPoint(
    #         positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         velocitys=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         accelerations=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         time_from_start=Duration(sec=1)
    #     ),
    #     JointTrajectoryPoint(
    #         positions=[1/24, 1/24, 1/24, 1/24, 1/24, 1/24],
    #         velocitys=[1/8, 1/8, 1/8, 1/8, 1/8, 1/8],
    #         accelerations=[1/4, 1/4, 1/4, 1/4, 1/4, 1/4],
    #         time_from_start=Duration(sec=1)
    #     ),
    # ]
    points = [create_trajectory_point(t, trajectory_data[t]) for t in range(1, 8)]

    # 创建轨迹客户端
    trajectory_client = TrajectoryClient(node)

    # 发送目标
    future = trajectory_client.send_goal(joint_names, points)

    if future:
        rclpy.spin_until_future_complete(node, future)
        # result = future.result()
        # if result:
        #     node.get_logger().info(f'Result: {result.result}')
        # else:
        #     node.get_logger().error('Failed to get result from action server.')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    

    # 测试：生成时间点为7的轨迹点
    # points = [create_trajectory_point(t, trajectory_data[t]) for t in range(1, 8)]
    # print(points)
