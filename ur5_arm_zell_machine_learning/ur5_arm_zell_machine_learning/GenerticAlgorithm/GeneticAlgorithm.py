import h5py
import numpy as np
from dataclasses import dataclass


@dataclass
class Individual:
    """
    p1(3, 16)->W1
    q1(4, 16)->W2
    p2(3, 16)->W3
    q2(4, 16)->W4  96---W7--64---W8---32---W9---1
    DH(4, 16)->W5
    id(1, 16)->W6
    """
    id: int
    age: int
    gene_matrices: list[np.ndarray]  # 基因矩阵列表
    species: str="IK_group1"


class GeneticAlogrithmALPS:
    def __init__(self) -> None:
        self.layer_number = 0
        self.population = {
            "L1": []
        }

    def sigmiod(self, x) -> np.ndarray:
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-x))

    def forward(self, input: np.ndarray, layer: str, id: int) -> np.ndarray:
        """
        前向传播计算
        :param input: 输入向量，形状必须为 (1, 19)
        :param layer: 种群层名（如 "L1"）
        :param id: 个体ID
        :return: 输出结果，形状为 (1, 1)
        """
        # 1. 获取个体
        individual = next(
            (ind for ind in self.population.get(layer, []) if ind.id == id),
            None
        )
        if individual is None:
            raise ValueError(f"个体不存在: layer={layer}, id={id}")

        # 2. 输入验证
        if input.shape != (1, 19):
            raise ValueError(f"输入形状需为 (1, 19)，当前为 {input.shape}")

        # 3. 定义分段规则（与基因矩阵前6个的行数对应）
        split_indices = [3, 4, 3, 4, 4, 1]  # 3+4+3+4+4+1=19
        if sum(split_indices) != input.shape[1]:
            raise ValueError("分段规则与输入维度不匹配")

        # 4. 分段计算
        x_parts = []
        start = 0
        for i, (n_input, matrix) in enumerate(zip(split_indices, individual.gene_matrices[:6])):
            end = start + n_input
            # 验证基因矩阵形状是否匹配
            if matrix.shape != (n_input, 16):
                raise ValueError(f"基因矩阵 {i} 形状应为 ({n_input}, 16)，实际为 {matrix.shape}")
            # 取输入片段并计算
            x_slice = input[:, start:end]  # (1, n_input)
            x_part = self.sigmiod(x_slice @ matrix)  # (1, 16)
            x_parts.append(x_part)
            start = end

        # 5. 拼接结果 (1, 16*6) = (1, 96)
        x = np.hstack(x_parts)  # 水平拼接

        # 6. 继续后续矩阵计算（索引6~8）
        for weight_matrix in individual.gene_matrices[6:]:
            x = self.sigmiod(x @ weight_matrix)

        return x  # 最终形状 (1, 1)
        
    def add_individual(self, layer: str, individual: Individual) -> None:
        self.population.setdefault(layer, []).append(individual)

    def generate_random_individual(self, layer: str, group: str = "IK_group1") -> None:
        individual = Individual(
            id = 0,
            age = 0,
            species = group,
            gene_matrices=[np.random.random((3,16)), np.random.random((4,16)),
                           np.random.random((3,16)), np.random.random((4,16)),
                           np.random.random((4,16)),np.random.random((1,16)),
                           np.random.random((96,64)),np.random.random((64,32)),
                           np.random.random((32,1))]
        )
        if layer in self.population:
            individual.id = len(self.population[layer])
            self.population[layer].append(individual)
        else:
            self.population[layer] = [individual]

    def save_population(self, file_path: str) -> None:
        """
        将 population 保存到 HDF5 文件
        文件结构示例：
            |- L1
                |- individual_0
                    |- gene_0 (dataset: shape=(3,16))
                    |- gene_1 (dataset: shape=(4,16))
                    ...
                    |- attributes (id, age, species)
                |- individual_1
                ...
            |- L2
            ...
        """
        with h5py.File(file_path, 'w') as f:
            for layer_name, individuals in self.population.items():
                # 为每层创建组
                layer_grp = f.create_group(layer_name)
                
                for idx, individual in enumerate(individuals):
                    # 为每个个体创建子组
                    ind_grp = layer_grp.create_group(f"individual_{idx}")
                    
                    # 保存基因矩阵
                    for gene_idx, matrix in enumerate(individual.gene_matrices):
                        ind_grp.create_dataset(f"gene_{gene_idx}", data=matrix)
                    
                    # 保存元数据
                    ind_grp.attrs["id"] = individual.id
                    ind_grp.attrs["age"] = individual.age
                    ind_grp.attrs["species"] = individual.species

    def load_population(self, file_path: str) -> None:
        """
        从 HDF5 文件加载 population 数据
        注意：会覆盖当前内存中的 population
        """
        self.population = {}  # 清空现有数据
        
        with h5py.File(file_path, 'r') as f:
            for layer_name in f:
                layer_grp = f[layer_name]
                individuals = []
                
                for ind_name in layer_grp:
                    ind_grp = layer_grp[ind_name]
                    
                    # 加载基因矩阵（按序号排序）
                    gene_matrices = []
                    gene_count = len([k for k in ind_grp.keys() if k.startswith("gene_")])
                    for gene_idx in range(gene_count):
                        gene_matrices.append(ind_grp[f"gene_{gene_idx}"][:])
                    
                    # 创建 Individual 对象
                    individual = Individual(
                        id=int(ind_grp.attrs["id"]),
                        age=int(ind_grp.attrs["age"]),
                        species=str(ind_grp.attrs["species"]),
                        gene_matrices=gene_matrices
                    )
                    
                    individuals.append(individual)
                
                self.population[layer_name] = individuals
    


network = GeneticAlogrithmALPS()
network.generate_random_individual("L1")
# for i in range(0,9):
#     print(network.population['L1'][0].gene_matrices[i])

print(network.forward(np.zeros([1, 19]), 'L1', 0))
# network.generate_random_individual("L1")
# network.save_population("population_data.h5")

# new_network = IKNetWork()
# new_network.load_population("population_data.h5")
# print(len(new_network.population["L1"]))  # 输出 2
# print(new_network.population['L1']) # 个体50kb级别数据量
    

