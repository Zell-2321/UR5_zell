import numpy as np
from typing import Union, List, Tuple

class ForwardKinematics:
    def __init__(self, DH: dict) -> None:
        """
        正向运动学类, 基于DH参数计算机器人变换矩阵
        Forward kinematics class, calculates transformation matrices based on DH parameters

        Args:
            DH (dict): DH参数字典, 必须包含'd', 'theta', 'a', 'alpha'四个键，每个键对应长度相同的列表
                      DH parameters dictionary, must contain keys: 'd', 'theta', 'a', 'alpha' 
                      with equal-length lists as values

        Raises:
            KeyError: 当缺少必要的DH参数键时触发 / Triggered when missing required DH parameter keys
            ValueError: 当DH参数长度不一致时触发 / Triggered when DH parameters have inconsistent lengths
        """
        # 检查必须的键是否存在 Check required keys
        required_keys = {'d', 'theta', 'a', 'alpha'}
        if not required_keys.issubset(DH.keys()):
            raise KeyError(f"DH table must contain keys: {required_keys}")

        # 验证所有参数长度一致 Validate parameter lengths consistency
        lengths = [len(v) for v in DH.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All DH parameters must have the same length")
        
        self.DH_table = DH  # 存储DH参数 Store DH parameters
        self.joint_number = lengths[0]  # 关节数量 Number of joints
        self.joint_value = np.zeros(self.joint_number)  # 初始化关节角度 Initialize joint angles
    
    def cal_transform_matrix(self, joint_id: int, joint_value: float, joint_type: str = 'revolute') -> np.ndarray:
        """
        计算单个关节的变换矩阵
        Calculate transformation matrix for a single joint

        Args:
            joint_id (int): 关节ID(1~n) / Joint ID (1~n)
            joint_value (float): 关节角度（旋转）或位移（平移） / Joint angle (revolute) or displacement (prismatic)
            joint_type (str): 关节类型，'revolute'（旋转）或'prismatic'（平移） / Joint type: 'revolute' or 'prismatic'

        Returns:
            np.ndarray: 4x4齐次变换矩阵 / 4x4 homogeneous transformation matrix

        Raises:
            IndexError: 当关节ID超出范围时触发 / Triggered when joint ID is out of range
            ValueError: 当关节类型非法时触发 / Triggered for invalid joint type
        """
        # 输入验证 Input validation
        if joint_id < 1 or joint_id > self.joint_number:
            raise IndexError(f"Joint ID must be in [1, {self.joint_number}], got {joint_id}")
        joint_idx = joint_id - 1  # 转换为0-based索引 Convert to 0-based index
        
        if joint_type not in ('revolute', 'prismatic'):
            raise ValueError(f"Invalid joint_type: {joint_type}")

        # 获取DH参数 Get DH parameters
        a = self.DH_table['a'][joint_idx]          # 连杆长度 Link length
        alpha = self.DH_table['alpha'][joint_idx]  # 连杆扭角 Link twist

        # 根据关节类型选择变量参数 Select variable parameter based on joint type
        if joint_type == 'revolute':
            theta = joint_value  # 旋转关节角度 Revolute joint angle
            d = self.DH_table['d'][joint_idx]  # 固定偏移量 Fixed offset
        else:
            d = joint_value      # 平移关节位移 Prismatic joint displacement
            theta = self.DH_table['theta'][joint_idx]  # 固定角度 Fixed angle

        # 预计算三角函数 Precompute trigonometric values
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        # 构建变换矩阵 Construct transformation matrix
        return np.array([
            [ct,    -st*ca,    st*sa,   a*ct],  # X轴分量 X-axis components
            [st,     ct*ca,   -ct*sa,   a*st],  # Y轴分量 Y-axis components
            [0,         sa,       ca,      d],  # Z轴分量 Z-axis components
            [0,          0,        0,      1]   # 齐次坐标 Homogeneous coordinate
        ])
        
    def cal_transform_matrix_range(self, lower: int, upper: int) -> np.ndarray:
        """
        计算从lower到upper关节的累积变换矩阵
        Calculate cumulative transformation matrix from lower to upper joints

        Args:
            lower (int): 起始关节索引(0-based) / Start joint index (0-based)
            upper (int): 结束关节索引(上限,0-based) / End joint index (exclusive, 0-based)

        Returns:
            np.ndarray: 累积的4x4变换矩阵 / Cumulative 4x4 transformation matrix

        Raises:
            IndexError: 当索引范围非法时触发 / Triggered for invalid index range
        """
        if not (0 <= lower < upper <= self.joint_number):
            raise IndexError(f"Invalid range: [{lower}, {upper}]")
        
        matrix = np.eye(4)  # 初始化单位矩阵 Initialize identity matrix
        for i in range(lower, upper):
            # 矩阵连乘累计变换 Matrix multiplication for cumulative transformation
            matrix = matrix @ self.cal_transform_matrix(i + 1, self.joint_value[i])
        return matrix
    
    def cal_ee_pos(self) -> np.ndarray:
        """
        计算机器人末端执行器位姿
        Calculate end-effector pose

        Returns:
            np.ndarray: 末端执行器的4x4位姿矩阵 / 4x4 pose matrix of end-effector
        """
        matrix = self.cal_transform_matrix_range(0, self.joint_number)
        return matrix
    
    def set_joint_angle(self, joint_angle: Union[np.ndarray, List, Tuple, float]) -> None:
        """
        设置关节角度值
        Set joint angles

        Args:
            joint_angle: 关节角度值，支持多种输入格式
                        Joint angles in numpy array, list, tuple or scalar

        Raises:
            ValueError: 当输入形状不匹配时触发 / Triggered when input shape doesn't match
        """
        # 转换为numpy数组并验证形状 Convert to numpy array and validate shape
        joint_angle = np.asarray(joint_angle, dtype=np.float64)
        if joint_angle.shape != (self.joint_number,):
            raise ValueError(f"Expected shape ({self.joint_number},), got {joint_angle.shape}")
        self.joint_value = joint_angle