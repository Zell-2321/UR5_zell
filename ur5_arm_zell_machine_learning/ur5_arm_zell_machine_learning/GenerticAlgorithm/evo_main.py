from Individual import *
from forward_kinematics import *
from Layer import *

def testIndividual():
    ind = Individual()
    print("ind.weights_tensor", ind.weights_tensor,"数据类型为：" ,type(ind.weights_tensor), "shape", ind.weights_tensor.shape)

    print(ind.age)
    ind.age += 1
    a = 2
    ind.age = a
    a += 1
    print(ind.age)
    print(type(torch.inf))

    total_weights = ind.weights_tensor.numel()
    print("total_weights:", total_weights)

    weights_buffer = torch.zeros(
            (1, total_weights),
            dtype=torch.float32,
            device="cpu"
        )
    ind_2 = Individual(weights_tensor=weights_buffer[0])
    new_weights_buffer = torch.ones(
            (1, 10),
            dtype=torch.float32,
            device="cpu"
        )
    ind_2.weights_tensor = new_weights_buffer[0]
    new_weights_buffer[0][0] = 3.0
    print(ind_2.weights_tensor, ind_2.weights_tensor.shape)

def testFK():
    DH_table_ur5 = {
        'd': [0.089159,  0.00000,  0.00000,  0.00000,  0.09465,  0.19145],
        'theta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'a': [0.00000, 0.42500, 0.39225,  0.00000,  0.00000,  0.0000],
        'alpha': [ -1.570796327, 0, 0, -1.570796327, 1.570796327, 0 ]
    }
    FK = ForwardKinematics(DH=DH_table_ur5)
    print("self.joint_number", FK.joint_number, "joint_value", FK.joint_value)

    # a = np.ones(6)
    # FK.joint_value = a
    # a[0] = 3.0
    print("FK.joint_value", FK.joint_value)

    print(FK.cal_transform_matrix(1, 0.0))

    print(FK.cal_transform_matrix_range(0, 6))

    print(FK.cal_ee_pos())

    FK.set_joint_angle(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]))

    print(FK.cal_ee_pos())

def testFC():
    DH_table_ur5 = {
        'd': [0.089159,  0.00000,  0.00000,  0.00000,  0.09465,  0.19145],
        'theta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'a': [0.00000, 0.42500, 0.39225,  0.00000,  0.00000,  0.0000],
        'alpha': [ -1.570796327, 0, 0, -1.570796327, 1.570796327, 0 ]
    }
    target_pos = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.19145],
        [0.0, 1.0, 0.0, 1.001059],
        [0.0, 0.0, 0.0, 1.0]
    ])
    ind = Individual()

    FK = ForwardKinematics(DH=DH_table_ur5)
    FC = FitnessCalculater(target_pos=target_pos, FK=FK)
    FC.calculate_fitness(individual=ind)
    fitness = ind.fitness
    # print(fitness)

def testLayer():
    DH_table_ur5 = {
        'd': [0.089159,  0.00000,  0.00000,  0.00000,  0.09465,  0.19145],
        'theta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'a': [0.00000, 0.42500, 0.39225,  0.00000,  0.00000,  0.0000],
        'alpha': [ -1.570796327, 0, 0, -1.570796327, 1.570796327, 0 ]
    }
    target_pos = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.19145],
        [0.0, 1.0, 0.0, 1.001059],
        [0.0, 0.0, 0.0, 1.0]
    ])
    layer_config_list = [
        LayerConfig(
            max_age=1,
            capacity=2000,
            mutation_rate=0.5,
            fitness_requirement=-torch.inf,
            beta=0.5,
            alpha=1.0
        ),
        LayerConfig(
            max_age=20,
            capacity=20,
            mutation_rate=0.5,
            fitness_requirement=-1.0,
            beta=0.5,
            alpha=0.7
        )
    ]
    FK = ForwardKinematics(DH=DH_table_ur5)
    FC = FitnessCalculater(target_pos=target_pos, FK=FK)
    L1 = Layer(layer_id=1, config=layer_config_list[0], Fitness_calculater=FC, previous_layer=None)
    L2 = Layer(layer_id=2, config=layer_config_list[1], Fitness_calculater=FC, previous_layer=L1)
    L1.initialize()
    L2.initialize()
    L2.generation_number = 1

    # 初始化测试
    # for ind in L1.individuals:
    #     print("weight_tensor:", ind.weights_tensor)
    #     print(ind.fitness, ind.age)

    # print("__________________________________________________________")

    # for ind in L2.individuals:
    #     print("weight_tensor:", ind.weights_tensor)
    #     print(ind.fitness, ind.age)

    # print("__________________________________________________________")

    # 添加个体测试
    List_new = []
    for i in range(8):
        ind = Individual(age=100)
        FC.calculate_fitness(ind)
        List_new.append(ind)
    
    L1.add_individuals(List_new)

    # for ind in L2.individuals:
    #     print("weight_tensor:", ind.weights_tensor)
    #     print(ind.fitness, ind.age)

    print("__________________________________________________________")

    parents = L2._select_parents()
    for parent in parents:
        print(parent)

    L2.evolve()
    # print("L2:", L2.pass_list)
    L1.evolve()
    # print(L1.pass_list)
    L2.evolve()
    # print("L2:", L2.pass_list)
    L1.evolve()
    # print(L1.pass_list)
    print(f"lenL1{len(L1.individuals)}, lenL2{len(L2.individuals)}")
        
def calculate_fitness(individual: Individual, target_pos, FK) -> None:
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
    
    for i in range(40):
        neural_network = NeuralNetwork(individual=individual)
        H_i_n = np.eye(4)
        H_0_i = FK.cal_ee_pos() # i=n

        for i in range(FK.joint_number, 0, -1):
            H_i_1_i = FK.cal_transform_matrix(joint_id=i, joint_value=FK.joint_value[i-1])
            H_i_1_target = H_i_1_i@np.linalg.inv(H_0_i)@target_pos
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
                torch.tensor([[FK.DH_table['d'][i-1]]], dtype=torch.float32),       # (1, 1)
                torch.tensor([[FK.joint_value[i-1]]], dtype=torch.float32),           # (1, 1) 注意这里可能去掉了多余的索引
                torch.tensor([[FK.DH_table['a'][i-1]]], dtype=torch.float32),       # (1, 1)
                torch.tensor([[FK.DH_table['alpha'][i-1]]], dtype=torch.float32),   # (1, 1)
                torch.tensor([[i]], dtype=torch.float32)                                  # (1, 1)
            ]
            input_tensor = torch.cat(components, dim=1)  # 最终形状 (1, 19)
            output = neural_network(input_tensor)  # 形状 (1, 1)
            FK.joint_value[i-1] = output[0][0]

            H_i_1_i = FK.cal_transform_matrix(joint_id=i, joint_value=FK.joint_value[i-1])
            H_i_n = H_i_1_i@H_i_n

        ee = FK.cal_ee_pos()
        R1, t1 = ee[:3, :3], ee[:3, 3]
        R2, t2 = target_pos[:3, :3], target_pos[:3, 3]
        # 计算相对旋转矩阵
        R_rel = np.dot(R1.T, R2)
        # 将相对旋转矩阵转换为轴角表示，提取旋转角度
        axis, angle= tf3d.axangles.mat2axangle(R_rel)
        theta = np.abs(angle)  # 旋转角度（弧度）
        # 旋转评分（角度越小分数越高）
        # 计算平移距离差
        d = np.linalg.norm(t2 - t1)
        print(f"轴角为：{axis},{theta}, 距离为{d}")

def testNN():
    DH_table_ur5 = {
        'd': [0.089159,  0.00000,  0.00000,  0.00000,  0.09465,  0.19145],
        'theta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'a': [0.00000, 0.42500, 0.39225,  0.00000,  0.00000,  0.0000],
        'alpha': [ -1.570796327, 0, 0, -1.570796327, 1.570796327, 0 ]
    }
    target_pos = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.19145],
        [0.0, 1.0, 0.0, 1.001059],
        [0.0, 0.0, 0.0, 1.0]
    ])
    FK = ForwardKinematics(DH=DH_table_ur5)
    file_path = f'best_individual_weights_layer_{4 + 1}.pth'
    weights_tensor = torch.load(file_path)
    loaded_individual = Individual()
    loaded_individual.weights_tensor = weights_tensor
    calculate_fitness(loaded_individual, target_pos, FK)
# testIndividual()
# testFK()
# testFC()
# testLayer()
testNN()


