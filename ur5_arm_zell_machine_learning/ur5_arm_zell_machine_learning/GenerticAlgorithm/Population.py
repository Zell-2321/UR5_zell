from Individual import *
from Layer import *
from NeuralNetwork import *

class Population:
    """ALPS管理的整体种群"""
    def __init__(self, layer_config, network_template):
        self.layers = [Layer(age) for age in layer_config]
        self.network_template = network_template  # 用于生成新个体的网络模板
        self.generation = 0
    
    def initialize_population(self, pop_size_per_layer):
        """初始化每层的种群"""
        for layer in self.layers:
            layer.population = [
                Individual(self._random_weights()) 
                for _ in range(pop_size_per_layer)
            ]
    
    def _random_weights(self):
        """生成随机权重（基于模板网络）"""
        net = self.network_template()
        net.apply(self._init_weights)
        return net.state_dict()
    
    def _init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)