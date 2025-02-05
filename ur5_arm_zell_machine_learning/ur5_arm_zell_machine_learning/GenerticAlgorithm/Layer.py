import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from Individual import *
import random

@dataclass
class LayerConfig:
    max_age: int          # 该层允许的最大年龄
    capacity: int         # 层容量（最大个体数）
    mutation_rate: float  # 该层的变异率
    fitness_requirement: float # 该层的适应度最低要求
    beta: float           # 该层的控制增长速度的系数
    alpha: float          # 从当前层选择父代的比例

class Layer:
    def __init__(self, layer_id, config: LayerConfig, device: str = "cpu", previous_layer: "Layer" = None) -> None:
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
        self.individual_number = 0
        self.generation_number = 0
        
        # 预分配内存优化（针对大规模种群）
        self._weights_buffer = None  # 张量缓存 [capacity, total_weights_dim]
        self._age_buffer = None      # 年龄张量 [capacity]
        self._fitness_buffer = None  # 适应度张量 [capacity]

    def initialize(self, template_individual: Individual) -> None:
        """
        初始化层的存储空间
        :param template_individual: 用于获取权重维度的模板个体
        重置 self.individuals列表
        设置个体数为0 (1层为层容量)
        重置缓存空间

        """
        total_weights = template_individual.weights_tensor.numel()
        
        # 预分配连续内存
        self._weights_buffer = torch.randn(
            (self.config.capacity, total_weights),
            dtype=torch.float32,
            device=self.device
        )

        self._age_buffer = torch.ones(
            self.config.capacity, 
            dtype=torch.int32,
            device=self.device
        )
        self._fitness_buffer = torch.full(
            (self.config.capacity,),
            -np.inf,
            dtype=torch.float32,
            device=self.device
        )

        self.individual_number = 0
        self.generation_number = 0
        
        if self.layer_id == 1:
            self.individual_number = self.config.capacity
        
        self.individuals = []
        # 初始化个体
        for i in range(self.config.capacity):
            ind = Individual()
            self._weights_buffer[i] = ind.weights_tensor
            self.individuals.append(ind)
            # TODO:更新所有个体Fitness

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
        empty_slots = torch.where(self._fitness_buffer == -np.inf)[0]
        
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