from Individual import *
from Layer import *
from NeuralNetwork import *

class Population:
    """ALPS管理的整体种群"""
    def __init__(self, layer_number: int, layer_config_list: List[LayerConfig], fitness_calculater: FitnessCalculater, device: str = "cpu"):
        self.Layers: List[Layer] = []
        self.layer_number = layer_number
        self.layer_config_list = layer_config_list
        self.device = device
        self.recycle_count = 0
        self.age_gap = layer_config_list[0].max_age
        self.fitness_calculater = fitness_calculater

    def initialize_population(self):
        if not len(self.layer_config_list) == self.layer_number:
            raise ValueError(f"配置列表长度与Layer层数设置不匹配 {self.layer_number},{len(self.layer_config_list)}")

        for i in range(self.layer_number):
            if i == 0:
                self.Layers.append(Layer(layer_id=i+1, config=self.layer_config_list[i], Fitness_calculater=self.fitness_calculater, device=self.device, previous_layer=None))
            else:
                self.Layers.append(Layer(layer_id=i+1, config=self.layer_config_list[i], Fitness_calculater=self.fitness_calculater, device=self.device, previous_layer=self.Layers[i-1]))
            self.Layers[i].initialize()  
        
        for i in range(len(self.Layers)-1):
            self.Layers[i].next_layer_fitness = self.Layers[i+1].config.fitness_requirement

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
    def update_fitness(self):
        for layer in self.Layers:
            for ind in layer.individuals:
                self.fitness_calculater.calculate_fitness(ind)
    
    def save_network(self):
        # 遍历每一层
        for layer_idx, layer in enumerate(self.Layers):
            if len(layer.individuals) > 0:
                # 找到该层适应度最高的个体
                best_individual = max(layer.individuals, key=lambda ind: ind.fitness)

                # 打印最好的个体适应度
                print(f"Layer {layer_idx + 1} - Best Individual Fitness: {best_individual.fitness}")

                # 保存该层最好的个体权重
                torch.save(best_individual.weights_tensor, f'best_individual_weights_layer_{layer_idx + 1}.pth')
                print(f"Best individual's weights for Layer {layer_idx + 1} saved successfully!")
            else:
                print(f"No individuals in Layer {layer_idx + 1}.")



    def ALSPEvolution(self):
        """

        """
        # self.initialize_population()
        self.recycle_count = 0
        while True:
            for i in range(len(self.Layers) - 1, -1, -1):
                if i == 0 and self.recycle_count % self.age_gap == 0:
                    self.Layers[0].initialize()
                    self.save_network()

                self.Layers[i].evolve()
            # self.fitness_calculater.target_pos = self.new_target_pos()
            # self.update_fitness()
            # print("case update")
            self.recycle_count += 1
            new_list = []
            fitness_list = []
            for layer in self.Layers:
                new_list.append(len(layer.individuals))
                fitness_list.append(layer.average_fitness)
            print(new_list, self.recycle_count)
            print(fitness_list)


if __name__== '__main__':
    # 参数设置
    # ur5 机械臂 DH表格
    # DH_table_ur5 = {
    #     'd': [0.089159,  0.00000,  0.00000,  0.10915,  0.09465,  0.0823],
    #     'theta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     'a': [0.00000, -0.42500, -0.39225,  0.00000,  0.00000,  0.0000],
    #     'alpha': [ 1.570796327, 0, 0, 1.570796327, -1.570796327, 0 ]
    # } # 此表格为Modified DH表， UR5官方文档按照此规则建立坐标系。 下面的DH表为标准DH表格，坐标系建立与官方不同，但是保证了给定相同角度的情况下，机械臂构型一致，
    # 对于工具系，{x_Modified_ee -> z_Standard_ee, y_M_ee -> x_S_ee, z_M_ee -> y_S_ee}
    # torch.set_num_threads(2) 

    DH_table_ur5 = {
        'd': [0.089159,  0.00000,  0.00000,  0.00000,  0.09465,  0.19145],
        'theta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'a': [0.00000, 0.42500, 0.39225,  0.00000,  0.00000,  0.0000],
        'alpha': [ -1.570796327, 0, 0, -1.570796327, 1.570796327, 0 ]
    }
    # ALPS每一层参数设置：
    ur5_layer_config_List = [
        LayerConfig(
            max_age=5,  # 16, 36, 64, 100, 144, 441, np.inf
            capacity=2000,
            mutation_rate=0.6,
            fitness_requirement=-np.inf,
            beta=0.5,
            alpha=1.0
        ),
        LayerConfig(
            max_age=12,  # 16, 36, 64, 100, 144, 441, np.inf
            capacity=1000,
            mutation_rate=0.55,
            fitness_requirement=0.0,
            beta=0.5,
            alpha=0.3
        ),
        LayerConfig(
            max_age=15,  # 16, 36, 64, 100, 144, 441, np.inf
            capacity=1000,
            mutation_rate=0.40,
            fitness_requirement=1.0,
            beta=0.5,
            alpha=0.3
        ),
        LayerConfig(
            max_age=18,  # 16, 36, 64, 100, 144, 441, np.inf
            capacity=1000,
            mutation_rate=0.33,
            fitness_requirement=3.5,
            beta=0.5,
            alpha=0.3
        ),
        LayerConfig(
            max_age=30,  # 16, 36, 64, 100, 144, 441, np.inf
            capacity=1000,
            mutation_rate=0.15,
            fitness_requirement=7.7,
            beta=0.5,
            alpha=0.3
        ),
        LayerConfig(
            max_age=50,  # 16, 36, 64, 100, 144, 441, np.inf
            capacity=1000,
            mutation_rate=0.05,
            fitness_requirement=10.0,
            beta=0.5,
            alpha=0.3
        ),LayerConfig(
            max_age=np.inf,  # 16, 36, 64, 100, 144, 441, np.inf
            capacity=1000,
            mutation_rate=0.10,
            fitness_requirement=3.0,
            beta=0.5,
            alpha=0.3
        )
    ]

    # 工具类初始化
    target_transformation_matrix = target_pos = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.19145],
        [0.0, 1.0, 0.0, 1.001059],
        [0.0, 0.0, 0.0, 1.0]
    ])

    forward_kinematics_class = ForwardKinematics(DH=DH_table_ur5)
    fitness_calculater = FitnessCalculater(target_pos=target_transformation_matrix, FK=forward_kinematics_class)

    # 创建种群
    ALPS_population = Population(layer_number=7, layer_config_list=ur5_layer_config_List, fitness_calculater=fitness_calculater, device="cpu")
    ALPS_population.initialize_population()
    # print(ALPS_population.Layers[0].)
    ALPS_population.ALSPEvolution()