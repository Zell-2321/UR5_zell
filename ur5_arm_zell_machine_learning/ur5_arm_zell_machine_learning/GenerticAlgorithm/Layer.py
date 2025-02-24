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

    def new_target_pos(self) -> np.ndarray:
        # 生成随机旋转矩阵
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)  # 归一化为单位向量
        angle = np.random.uniform(0, 2 * np.pi)  # 随机旋转角度
        R = tf3d.axangles.axangle2mat(axis, angle)  # 旋转矩阵 (3x3)

        # 生成随机平移向量，模长不超过 1
        t = np.random.uniform(-1, 1, size=3)
        if np.linalg.norm(t) > 1:
            t = t / np.linalg.norm(t)  # 归一化后再缩放
            t *= np.random.uniform(0, 1)  # 缩放到 [0,1] 范围内

        # 构造齐次变换矩阵 (4x4)
        H = np.eye(4)
        H[:3, :3] = R  # 旋转部分
        H[:3, 3] = t   # 平移部分

        return H  

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
        
        # self.target_pos = self.new_target_pos()
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
            if fitness_now - fitness_prev <0:
                fitness += (fitness_now-fitness_prev) * 5
            fitness += (fitness_now-fitness_prev) * (11-i)
            fitness_prev = fitness_now
            # print("当前得分：", fitness)

            H_i_n = H_i_1_i@H_i_n

        individual.fitness = torch.tensor(fitness/4, dtype=torch.float32)
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
        self.generation_number = 0
        self.fitness_calculater = Fitness_calculater
        self.logger = setup_logger(log_file=f"logs/Layer_Class_Layer{self.layer_id}.log")
        self.next_layer_fitness = torch.inf

    def initialize(self) -> None:
        """
        初始化层的存储空间
        :param template_individual: 用于获取权重维度的模板个体
        重置 self.individuals列表
        设置个体数为0 (1层为层容量)
        重置缓存空间

        """
        self.generation_number = 0
        self.individuals = []
        # 初始化个体
        if self.layer_id == 1:
            for _ in range(self.config.capacity):
                ind = Individual()
                self.individuals.append(ind)
                self.fitness_calculater.calculate_fitness(ind) 

    def add_individuals(self, individuals: List[Individual]) -> None:
        """
        添加新个体到本层（来自上层迁移或初始化）
        :param individuals: 待添加的个体列表
        """
        for i in range(len(individuals)):
            # 判断个体fitness是否达到当前层数要求
            if individuals[i].fitness >= self.next_layer_fitness:
                self.pass_list.append(individuals[i])
                continue

            if individuals[i].fitness >= self.config.fitness_requirement:
                # 判断年龄是否满足要求：
                if individuals[i].age <= self.config.max_age:
                    # 存在空位
                    if len(self.individuals) < self.config.capacity:
                        self.individuals.append(individuals[i])
                    
                    # 当前层已满
                    else:
                        min_ind = min(self.individuals, key=lambda ind: ind.fitness)
                        # 若当前个体更优则添加
                        if individuals[i].fitness > min_ind.fitness:
                            self.individuals.remove(min_ind)
                            self.individuals.append(individuals[i])
                

    def evolve(self, selection_ratio: float = 0.4) -> None:
        """
        执行遗传操作（选择、交叉、变异）
        :param selection_ratio: 选择保留的优秀个体比例
        """
        # 0. 处理上层个体
        self.pass_list = [] # 清空当前层传递列表
        if self.previous_layer is not None:
            self.add_individuals(self.previous_layer.pass_list)
        
        # 1. 选择
        # if len(self.individuals) < 10:
        #     num_keep = len(self.individuals)
        # else:
        #     num_keep = int(len(self.individuals) * selection_ratio) # 保留的个体数

        selection_ratio = 1.0*np.exp(-len(self.individuals)/self.config.capacity)
        num_keep = int(len(self.individuals) * selection_ratio)

        sorted_indices = sorted(
            range(len(self.individuals)),                   # 生成索引序列
            key=lambda i: self.individuals[i].fitness,     # 按 fitness 作为排序键
            reverse=True                                   # 从大到小排序
        )
        elite_indices = sorted_indices[:num_keep]
        self.generation_number += 1

        if self.layer_id == 1:
            print("num_keep:", num_keep)
        
        # 2. 交叉（锦标赛选择）
        parents = self._select_parents(selection_ratio)
        if self.layer_id == 1:
            print("parent_len:", len(parents))
        
        # 3. 生成后代
        children = []
        for i in range(0, len(parents), 2):
            if i+1 >= len(parents):
                break
            for _ in range(self.layer_id):
                child = self._crossover(parents[i], parents[i+1])
                child = self._mutate(child)
                self.fitness_calculater.calculate_fitness(child)
                children.append(child)
            

        if self.layer_id == 1:
            print("children_len:", len(children))
        
        # 4. 更新种群
        new_population = [self.individuals[i] for i in elite_indices] + children
        self.individuals = []
        self.add_individuals(new_population)
    
    # def _select_parents(self, selection_ratio: float = 0.4) -> List[Individual]:
    #     """
    #     从本层以及上层的个体中通过锦标赛选择个体
    #     selection_ratio: 选择个体的比例
    #     """
    #     if len(self.individuals) < 10:
    #         num_keep = len(self.individuals)
    #     else:
    #         num_keep = int(len(self.individuals) * selection_ratio) # 保留的个体数

    #     # 锦标赛选择 父代数量为期望子代数量的2倍
    #     tournament_size = 3
    #     parents = []
    #     N_layer = int(self.config.capacity*(1-np.exp(-self.config.beta*self.generation_number))) # 当前层应该有的个体数

    #     num_new_individuals = max(0, N_layer - num_keep) # 需要生成的子代数
    #     num_current_layer_parents = 2*int(num_new_individuals * self.config.alpha)
    #     num_previous_layer_parents = 2*(num_new_individuals - num_current_layer_parents)

    #     # 当前层锦标赛选择
    #     for _ in range(num_current_layer_parents):
    #         tournament_contestants = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
    #         if len(tournament_contestants) > 0:
    #                 winner = max(tournament_contestants, key=lambda ind: ind.fitness)
    #                 parents.append(winner)

    #     # 上一层锦标赛选择
    #     for _ in range(num_previous_layer_parents):
    #         if self.previous_layer is not None:
    #             tournament_contestants = random.sample(self.previous_layer.individuals, min(tournament_size, len(self.previous_layer.individuals)))
    #             if len(tournament_contestants) > 0:
    #                 winner = max(tournament_contestants, key=lambda ind: ind.fitness)
    #                 parents.append(winner)
    #         else: 
    #             print(f"Layer{self.layer_id} can not get data from previous layer! Skip selection")

    #     random.shuffle(parents)

    #     return parents
    def _select_parents(self, selection_ratio: float = 0.1) -> List[Individual]:
        """
        从本层以及上层的个体中通过锦标赛选择个体
        selection_ratio: 选择个体的比例
        """
        if len(self.individuals) < 10:
            num_keep = len(self.individuals)
        else:
            num_keep = int(len(self.individuals) * selection_ratio)+10  # 保留的个体数
        if num_keep > len(self.individuals):
            num_keep = len(self.individuals)

        # 锦标赛选择，父代数量为期望子代数量的2倍
        tournament_size = 3
        parents = []
        selected_set = set()  # 用于跟踪已选择的个体，防止重复

        # N_layer = int(self.config.capacity * (1 - np.exp(-self.config.beta * self.generation_number)))  # 当前层应该有的个体数
        # num_new_individuals = max(0, N_layer - num_keep)  # 需要生成的子代数
        num_new_individuals = max(0, self.config.capacity - num_keep)  # 需要生成的子代数
        num_current_layer_parents = 2 * int(num_new_individuals * self.config.alpha)
        num_previous_layer_parents = 2 * (num_new_individuals - num_current_layer_parents)

        # 当前层锦标赛选择
        for _ in range(num_current_layer_parents):
            available_contestants = [ind for ind in self.individuals if ind not in selected_set]
            if len(available_contestants) == 0:
                break  # 没有更多可选的个体，跳出循环

            tournament_contestants = random.sample(available_contestants, min(tournament_size, len(available_contestants)))
            winner = max(tournament_contestants, key=lambda ind: ind.fitness)
            
            parents.append(winner)
            selected_set.add(winner)  # 标记为已选择


        # 上一层锦标赛选择
        for _ in range(num_previous_layer_parents):
            if self.previous_layer is not None:
                available_contestants_prev = [ind for ind in self.previous_layer.individuals if ind not in selected_set]
                if len(available_contestants_prev) == 0:
                    break

                tournament_contestants = random.sample(available_contestants_prev, min(tournament_size, len(available_contestants_prev)))
                winner = max(tournament_contestants, key=lambda ind: ind.fitness)
                
                parents.append(winner)
                selected_set.add(winner)  # 标记为已选择
            else:
                print(f"Layer {self.layer_id} cannot get data from the previous layer! Skipping selection.")

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
        
        return torch.mean(torch.tensor([ind.fitness.item() for ind in self.individuals]))

    # def to(self, device: str):
    #     """转移层数据到指定设备"""
    #     self.device = torch.device(device)
    #     self._weights_buffer = self._weights_buffer.to(device)
    #     self._age_buffer = self._age_buffer.to(device)
    #     self._fitness_buffer = self._fitness_buffer.to(device)
    #     return self