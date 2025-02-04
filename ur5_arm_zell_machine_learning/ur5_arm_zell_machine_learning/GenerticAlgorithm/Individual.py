from dataclasses import dataclass
import numpy as np
import torch
import time

@dataclass
class Individual:
    """
    个体：包含网络权重、年龄、适应度
    基因矩阵结构：
    p1(3, 16) -> W1
    q1(4, 16) -> W2
    p2(3, 16) -> W3
    q2(4, 16) -> W4   ---96 ---W7--64---W8---32---W9---1
    DH(4, 16) -> W5
    id(1, 16) -> W6
    """
    age: int = 0
    gene_shapes: tuple = ((3,16), (4,16), (3,16), (4,16), (4,16), (1,16), (96,64), (64,32), (32,1))  # 各层权重形状
    weights_tensor: torch.Tensor = None  # 展平后的权重张量（核心优化）
    fitness: float = -np.inf
    species: str = "IK_group1"  # 可保留为轻量级标签

    def __post_init__(self):
        if self.weights_tensor is None:
            self.weights_tensor = self._init_random_weights()
    
    def _init_random_weights(self) -> torch.Tensor:
        """按基因形状初始化随机权重（展平为1D张量）"""
        flattened = [np.random.randn(*s).ravel().astype(np.float32) for s in self.gene_shapes]
        return torch.from_numpy(np.concatenate(flattened))

    def get_gene_matrices(self) -> list[np.ndarray]:
        """将展平张量还原为原始矩阵列表（按gene_shapes切割）"""
        ptr = 0
        matrices = []
        for shape in self.gene_shapes:
            size = np.prod(shape)
            layer_data = self.weights_tensor[ptr:ptr+size].numpy().reshape(shape)
            matrices.append(layer_data)
            ptr += size
        return matrices

if __name__== '__main__':
    # (1) 创建随机权重的个体:-----------------------------------------------------------------------------------------
    # 默认使用 gene_shapes 和随机初始化权重
    individual = Individual()  # 自动调用 _init_random_weights

    # 查看展平后的权重张量
    print(individual.weights_tensor.shape)  # 输出总元素数：3*16 + 4*16 + ... + 32*1 = 计算总和
    print(individual.weights_tensor.dtype)  # torch.float32

    # 查看个体属性
    print(individual.age)       # 0（默认值）
    print(individual.fitness)   # -inf（默认未评估）
    print(individual.species)   # "IK_group1"


    # (2) 获取原始矩阵形式的权重 -----------------------------------------------------------------------------------------
    matrices = individual.get_gene_matrices()

    # 例如访问第一个权重矩阵 W1 (3x16)
    w1 = matrices[0]
    print(w1.shape)  # (3, 16)

    # 访问第七个权重矩阵 W7 (96x64)
    w7 = matrices[6]
    print(w7.shape)  # (96, 64)

    # 2. 自定义初始化
    # (1) 从已有权重加载 -----------------------------------------------------------------------------------------
    # 假设有一个展平后的权重张量（需与 total_size 匹配）
    total_size = sum(np.prod(s) for s in Individual.gene_shapes)
    custom_weights = torch.randn(total_size, dtype=torch.float32)

    # 创建自定义权重的个体
    custom_individual = Individual(
        weights_tensor=custom_weights,
        age=5,
        fitness=0.8,
        species="IK_group2"
    )
    # (2) 部分参数覆盖 -----------------------------------------------------------------------------------------
    # 仅修改 age 和 species，权重仍随机初始化
    individual = Individual(age=10, species="experimental_group")

    #  3. 典型应用场景
    # (1) 遗传算法中的个体 -----------------------------------------------------------------------------------------
    # 创建初始种群
    population = [Individual() for _ in range(100)]


    # 评估适应度
    for ind in population:
        matrices = ind.get_gene_matrices()
        # 将矩阵输入到模型计算适应度...
        # ind.fitness = calculate_fitness(matrices)

    # 选择、交叉、变异操作...

    # (2) 神经网络权重管理 -----------------------------------------------------------------------------------------
    class MyModel(torch.nn.Module):
        def __init__(self, individual: Individual):
            super().__init__()
            matrices = individual.get_gene_matrices()
            self.w1 = torch.nn.Parameter(torch.tensor(matrices[0]))
            self.w2 = torch.nn.Parameter(torch.tensor(matrices[1]))
            # 其他层...

    # 将个体权重加载到模型
    individual = Individual()
    model = MyModel(individual)

    # 4. 结合文件加载（扩展场景） -----------------------------------------------------------------------------------------
    # 假设已有 PopulationLayer 类（见前文代码）
    # layer = PopulationLayer("weights.bin", gene_shapes=Individual.gene_shapes)

    # 从文件加载第3个个体
    # individual_3 = layer.load_individual(offset=3)

    # 使用个体
    # matrices = individual_3.get_gene_matrices()
