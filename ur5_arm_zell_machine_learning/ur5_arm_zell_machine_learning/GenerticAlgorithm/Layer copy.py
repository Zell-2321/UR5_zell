import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from Individual import *
import random
from forward_kinematics import *
from NeuralNetwork import *
import transforms3d as tf3d
from Logger import *

@dataclass
class LayerConfig:
    max_age: int          # 该层允许的最大年龄
    capacity: int         # 层容量（最大个体数）
    mutation_rate: float  # 该层的变异率
    fitness_requirement: float # 该层的适应度最低要求
    beta: float           # 该层的控制增长速度的系数
    alpha: float          # 从当前层选择父代的比例

class FitnessCalculater:
    def __init__(self, target_pos: np.ndarray, FK: ForwardKinematics) -> None:
        # self.individual = individual
        self.target_pos = target_pos
        self.test_case = None
        self.FK = FK

    def calculate_fitness(self, individual: Individual) -> None:
        """
        Fitness变量: fitness: 函数返回值， fitness_prev = 运动前fitness, fitness_now: 运动后 fitness
            fitness += fitness_now - fitness_prev

        初始化准备：
            fitness = 0.0, 计算fitness_prev, fitness_now = 0.0
            H_i^n = I
            H_0^i = H_0^n = ee_pos
        从最远段关节开始，到第一个关节 for i in range(n, 0, -1):
            1. 计算网络输入: H_i^n(I),  H_{i-1}^target
                H_{i-1}^i 由DH表得到 
                H_{i-1}^target = H_{i-1}^{0} * H_{0}^{target} = H_{i-1}^i * inv(H_0^i) * H_0^target(self.target_pos)
                H_0^i = H_0^i * inv(H_{i-1}^{i})

            2. 输入Network计算下一时刻关节角, 更新关节角列表， 计算新的fitness_now, fitness += (fitness_now - fitness_prev) * (11-i) (越往后应该越接近目标位置)
                
            3. fitness_prev = fitness_now

            4. 更新变量
                H_{i-1}^{i} = 由新的角度计算出的矩阵
                H_i^n = inv(H_{i-1}^{i}) * H_i^n
        """
        
        fitness = 0.0
        fitness_prev = 0.0
        fitness_now = 0.0
        neural_network = NeuralNetwork(individual=individual)
        H_i_n = np.eye(4)
        H_0_i = self.FK.cal_ee_pos() # i=n

        for i in range(self.FK.joint_number, 0, -1):
            H_i_1_i = self.FK.cal_transform_matrix(joint_id=i, joint_value=self.FK.joint_value[i-1])
            H_i_1_target = H_i_1_i@np.linalg.inv(H_0_i)@self.target_pos

            fitness_prev = self.evaluate_transform_similarity(H_i_1_target, H_i_1_i@H_i_n)

            H_0_i = H_0_i@np.linalg.inv(H_i_1_i)

            # 计算输入数值
            quat_1 = tf3d.quaternions.mat2quat(H_i_n[:3, :3])
            pos_1 = H_i_n[:3, 3]
            quat_2 = tf3d.quaternions.mat2quat(H_i_1_target[:3, :3])
            pos_2 = H_i_1_target[:3, 3]
            components = [
                torch.tensor(quat_1, dtype=torch.float32).view(1, -1),     # (1, 4)
                torch.tensor(pos_1, dtype=torch.float32).view(1, -1),      # (1, 3)
                torch.tensor(quat_2, dtype=torch.float32).view(1, -1),     # (1, 4)
                torch.tensor(pos_2, dtype=torch.float32).view(1, -1),      # (1, 3)
                torch.tensor([[self.FK.DH_table['d'][i-1]]], dtype=torch.float32),       # (1, 1)
                torch.tensor([[self.FK.joint_value[i-1]]], dtype=torch.float32),           # (1, 1) 注意这里可能去掉了多余的索引
                torch.tensor([[self.FK.DH_table['a'][i-1]]], dtype=torch.float32),       # (1, 1)
                torch.tensor([[self.FK.DH_table['alpha'][i-1]]], dtype=torch.float32),   # (1, 1)
                torch.tensor([[i]], dtype=torch.float32)                                  # (1, 1)
            ]
            input_tensor = torch.cat(components, dim=1)  # 最终形状 (1, 19)
            output = neural_network(input_tensor)  # 形状 (1, 1)
            self.FK.joint_value[i-1] = output[0][0]

            H_i_1_i = self.FK.cal_transform_matrix(joint_id=i, joint_value=self.FK.joint_value[i-1])

            fitness_now = self.evaluate_transform_similarity(H_i_1_target, H_i_1_i@H_i_n)
            fitness += (fitness_now-fitness_prev) * (11-i)
            fitness_prev = fitness_now
            # print("当前得分：", fitness)

            H_i_n = H_i_1_i@H_i_n

        individual.fitness = torch.tensor(fitness, dtype=torch.float32)
        ee = self.FK.cal_ee_pos()
        R1, t1 = ee[:3, :3], ee[:3, 3]
        R2, t2 = self.target_pos[:3, :3], self.target_pos[:3, 3]
        # 计算相对旋转矩阵
        R_rel = np.dot(R1.T, R2)
        # 将相对旋转矩阵转换为轴角表示，提取旋转角度
        axis, angle= tf3d.axangles.mat2axangle(R_rel)
        theta = np.abs(angle)  # 旋转角度（弧度）
        # 旋转评分（角度越小分数越高）
        # 计算平移距离差
        d = np.linalg.norm(t2 - t1)
        # print(f"轴角为：{axis},{theta}, 距离为{d}")
    
    def evaluate_transform_similarity(
        self,
        H1: np.ndarray, 
        H2: np.ndarray,
        rotation_weight: float = 0.3,
        translation_scale: float = 1.0
    ) -> float:
        """
        评估两个齐次变换矩阵的相似度（评分越高表示越接近）
        
        参数：
            H1, H2 (np.ndarray): 4x4 齐次变换矩阵
            rotation_weight (float): 旋转分量的权重（0~1）
            translation_scale (float): 平移分量的归一化尺度（单位与平移量一致）
        
        返回：
            float: 综合相似度评分（0~1）
        """
        # 参数检查
        if H1.shape != (4, 4) or H2.shape != (4, 4):
            raise ValueError("H1 和 H2 必须是 4x4 的齐次变换矩阵。")
        if not (0 <= rotation_weight <= 1):
            raise ValueError("rotation_weight 必须在 [0, 1] 之间。")
        if translation_scale <= 0:
            raise ValueError("translation_scale 必须为正数。")
        # 提取旋转矩阵和平移向量
        R1, t1 = H1[:3, :3], H1[:3, 3]
        R2, t2 = H2[:3, :3], H2[:3, 3]
        # 计算相对旋转矩阵
        R_rel = np.dot(R1.T, R2)
        # 将相对旋转矩阵转换为轴角表示，提取旋转角度
        axis, angle= tf3d.axangles.mat2axangle(R_rel)
        theta = np.abs(angle)  # 旋转角度（弧度）
        # 旋转评分（角度越小分数越高）
        rotation_score = 1 - (theta / np.pi)  # 归一化到 [0,1]
        # 计算平移距离差
        d = np.linalg.norm(t2 - t1)
        translation_score = np.exp(- (d ** 2) / (2 * translation_scale ** 2))  # 高斯衰减
        # 综合评分（加权平均）
        total_score = (
            rotation_weight * rotation_score +
            (1 - rotation_weight) * translation_score
        )
        return np.clip(total_score, 0.0, 1.0)

class Layer:
    def __init__(self, layer_id, config: LayerConfig, Fitness_calculater = FitnessCalculater, device: str = "cpu", previous_layer: "Layer" = None) -> None:
        """
        ALPS年龄层实现
        :param config: 层配置参数
        :param device: 存储设备 (cpu/cuda)
        """
        self.layer_id = layer_id
        self.config = config
        self.device = torch.device(device)
        self.individuals: List[Individual] = [] # 对象索引列表
        self.pass_list: List[Individual] = [] # 传递给下层的对象列表
        self.previous_layer = previous_layer
        # self.individual_number = 0
        self.generation_number = 0
        self.fitness_calculater = Fitness_calculater
        self.logger = setup_logger(log_file=f"logs/Layer_Class_Layer{self.layer_id}.log")
        
        # 预分配内存优化（针对大规模种群）
        # self._weights_buffer = None  # 张量缓存 [capacity, total_weights_dim]
        # self._age_buffer = None      # 年龄张量 [capacity]
        # self._fitness_buffer = None  # 适应度张量 [capacity]

    def initialize(self, template_individual: Individual) -> None:
        """
        初始化层的存储空间
        :param template_individual: 用于获取权重维度的模板个体
        重置 self.individuals列表
        设置个体数为0 (1层为层容量)
        重置缓存空间

        """
        # total_weights = template_individual.weights_tensor.numel()
        
        # 预分配连续内存
        # self._weights_buffer = torch.randn(
        #     (self.config.capacity, total_weights),
        #     dtype=torch.float32,
        #     device=self.device
        # )

        # self._age_buffer = torch.ones(
        #     self.config.capacity, 
        #     dtype=torch.int32,
        #     device=self.device
        # )
        # self._fitness_buffer = torch.full(
        #     (self.config.capacity,),
        #     -torch.inf,
        #     dtype=torch.float32,
        #     device=self.device
        # )

        # self.individual_number = 0
        self.generation_number = 0
        
        # if self.layer_id == 1:
            # self.individual_number = self.config.capacity
        
        self.individuals = []
        # 初始化个体
        for i in range(self.config.capacity):
            ind = Individual()
            # self._weights_buffer[i] = ind.weights_tensor
            self.individuals.append(ind)
            # TODO:更新所有个体Fitness
            self.fitness_calculater.calculate_fitness(ind) 
            # self._fitness_buffer[i] = ind.fitness #拷贝


    def get_migrants(self) -> List[Individual]:
        """获取需要迁移到下一层的个体"""
        age_mask = self._age_buffer > self.config.max_age
        migrant_indices = torch.where(age_mask)[0].tolist()
        
        migrants = []
        for idx in migrant_indices:
            # 更新个体元数据
            self.individuals[idx].age = self._age_buffer[idx].item()
            self.individuals[idx].fitness = self._fitness_buffer[idx].item()
            migrants.append(self.individuals[idx])
        
        # 清空迁移个体的位置
        self._weights_buffer[age_mask] = 0
        self._fitness_buffer[age_mask] = -np.inf
        return migrants

    def add_individuals(self, individuals: List[Individual]) -> None:
        """
        添加新个体到本层（来自上层迁移或初始化）
        :param individuals: 待添加的个体列表
        """
        print(self._fitness_buffer)
        print(self._fitness_buffer == -torch.inf)
        empty_slots = torch.where(self._fitness_buffer == -torch.inf)[0]
        
        for i in range(len(individuals)):
            # 判断个体fitness是否达到当前层数要求
            if individuals[i].fitness >= self.config.fitness_requirement:
                # 判断年龄是否满足要求：
                if individuals[i].age >= self.config.max_age:
                    # 存在空位
                    if i < len(empty_slots):
                        self._weights_buffer[empty_slots[i]] = individuals[i].weights_tensor.to(self.device)
                        self._age_buffer[empty_slots[i]] = individuals[i].age
                        self._fitness_buffer[empty_slots[i]] = individuals[i].fitness
                        self.individuals[empty_slots[i]] = individuals[i]
                        self.individual_number += 1
                    
                    # 当前层已满
                    else:
                        min_index = torch.argmin(self._fitness_buffer)
                        # 若当前个体更优则添加
                        if individuals[i].fitness > self._fitness_buffer[min_index]:
                            self._weights_buffer[min_index] = individuals[i].weights_tensor.to(self.device)
                            self._age_buffer[min_index] = individuals[i].age
                            self._fitness_buffer[min_index] = individuals[i].fitness
                            self.individuals[min_index] = individuals[i]
                
                else:
                    self.pass_list.append(individuals[i])

    def evolve(self, selection_ratio: float = 0.2) -> None:
        """
        执行遗传操作（选择、交叉、变异）
        :param selection_ratio: 选择保留的优秀个体比例
        """
        # 0. 处理上层个体
        self.pass_list = [] # 清空当前层传递列表
        self.add_individuals(self.previous_layer.pass_list)

        # 1. 选择
        num_keep = int(self.individual_number * selection_ratio) # 保留的个体数
        sorted_indices = torch.argsort(self._fitness_buffer, descending=True)
        elite_indices = sorted_indices[:num_keep]
        self.generation_number += 1
        
        # 2. 交叉（锦标赛选择）
        parents = self._select_parents(selection_ratio)
        
        # 3. 生成后代
        children = []
        for i in range(0, len(parents), 2):
            if i+1 >= len(parents):
                break
            child = self._crossover(parents[i], parents[i+1])
            child = self._mutate(child)
            children.append(child)
            # TODO: 计算fitness
            self.fitness_calculater.calculate_fitness()
        
        # 4. 更新种群
        new_population = [self.individuals[i] for i in elite_indices] + children
        self.initialize(Individual())
        self.add_individuals(new_population)
    
    def _select_parents(self, selection_ratio: float = 0.2) -> List[Individual]:
        """
        从本层以及上层的个体中通过锦标赛选择个体
        selection_ratio: 选择个体的比例
        """
        num_keep = int(self.individual_number * selection_ratio) # 保留的个体数
        # sorted_indices = torch.argsort(self._fitness_buffer, descending=True)
        # elite_indices = sorted_indices[:num_keep]

        # 锦标赛选择 父代数量为期望子代数量的2倍
        tournament_size = 3
        parents = []
        N_layer = int(self.config.capacity*(1-np.exp(-self.config.beta*self.generation_number))) # 当前层应该有的个体数

        num_new_individuals = max(0, N_layer - num_keep) # 需要生成的子代数
        num_current_layer_parents = 2*int(num_new_individuals * self.config.alpha)
        num_previous_layer_parents = 2*(num_new_individuals - num_current_layer_parents)

        valid_indices_current_layer = torch.nonzero(self._fitness_buffer != -np.inf).flatten()
        valid_indices_previous_layer = torch.nonzero(self.previous_layer._fitness_buffer != -np.inf).flatten()

        # 当前层锦标赛选择
        for _ in range(num_current_layer_parents):
            candidates = torch.randint(0, valid_indices_current_layer.size(0), (tournament_size,))
            selected_candidates = valid_indices_current_layer[candidates]
            winner = selected_candidates[torch.argmax(self._fitness_buffer[selected_candidates])]
            parents.append(self.individuals[winner])

        # 上一层锦标赛选择
        for _ in range(num_previous_layer_parents):
            candidates = torch.randint(0, valid_indices_previous_layer.size(0), (tournament_size,))
            selected_candidates = valid_indices_previous_layer[candidates]
            winner = selected_candidates[torch.argmax(self.previous_layer._fitness_buffer[selected_candidates])]
            parents.append(self.previous_layer.individuals[winner])

        random.shuffle(parents)

        return parents

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """均匀交叉（基于缓冲区加速）"""
        mask = torch.rand_like(parent1.weights_tensor) < 0.5
        child_weights = torch.where(mask, parent1.weights_tensor, parent2.weights_tensor)
        return Individual(weights_tensor=child_weights, age=max(parent1.age, parent2.age))

    def _mutate(self, individual: Individual) -> Individual:
        """高斯变异（向量化实现）"""
        noise = torch.randn_like(individual.weights_tensor) * 0.1
        mask = torch.rand_like(individual.weights_tensor) < self.config.mutation_rate
        mutated = individual.weights_tensor + mask * noise
        return Individual(weights_tensor=mutated, age=individual.age+1)

    @property
    def average_fitness(self) -> float:
        """计算本层的平均适应度"""
        valid_mask = self._fitness_buffer != -np.inf
        return self._fitness_buffer[valid_mask].mean().item()

    def to(self, device: str):
        """转移层数据到指定设备"""
        self.device = torch.device(device)
        self._weights_buffer = self._weights_buffer.to(device)
        self._age_buffer = self._age_buffer.to(device)
        self._fitness_buffer = self._fitness_buffer.to(device)
        return self