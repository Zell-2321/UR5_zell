import torch
import torch.nn as nn
import numpy as np
from Individual import *

class NeuralNetwork(nn.Module):
    def __init__(self, individual: Individual, device="cpu") -> None:
        """
        与遗传算法深度集成的神经网络
        :param individual: 关联的Individual实例
        :param device: 计算设备 (cpu/cuda)
        """
        super().__init__()
        self.individual = individual
        self.device = torch.device(device)
        self.split_indices = [3, 4, 3, 4, 4, 1]  # 输入分段规则
        
        # 预分割权重矩阵（提升前向传播效率）
        self.gene_matrices = self._parse_weights()
        
    def _parse_weights(self) -> list[torch.Tensor]:
        """将展平的weights_tensor解析为各层权重矩阵"""
        matrices = []
        ptr = 0
        
        # 解析前6个基因矩阵（对应输入分段）
        for shape in self.individual.gene_shapes[:6]:
            size = np.prod(shape)
            matrix = self.individual.weights_tensor[ptr:ptr+size].view(shape).to(self.device)
            matrices.append(matrix)
            ptr += size
        
        # 解析后续全连接层（W7-W9）
        for shape in self.individual.gene_shapes[6:]:
            size = np.prod(shape)
            matrix = self.individual.weights_tensor[ptr:ptr+size].view(shape).to(self.device)
            matrices.append(matrix)
            ptr += size
            
        return matrices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        分段式前向传播（支持批量计算）
        :param x: 输入张量，形状为 (batch_size, 19)
        :return: 输出张量，形状为 (batch_size, 1)
        """
        # 输入验证
        if x.shape[1] != 19:
            raise ValueError(f"输入特征维度应为19，当前为 {x.shape[1]}")
        
        # 转换为浮点型并发送到设备 TODO： 暂不支持GPU加速
        # x = x.float().to(self.device)
        
        # 分段计算（第一部分：基因矩阵1-6）
        parts = []
        start = 0
        for n_input, matrix in zip(self.split_indices, self.gene_matrices[:6]):
            end = start + n_input
            x_part = x[:, start:end]
            # 矩阵乘法并激活 (batch, n_input) @ (n_input, 16) -> (batch, 16)
            activated = torch.sigmoid(torch.mm(x_part, matrix))
            parts.append(activated)
            start = end
        
        # 拼接中间结果 (batch, 96)
        x = torch.cat(parts, dim=1)
        
        # 后续全连接计算（基因矩阵7-9）
        for matrix in self.gene_matrices[6:]:
            x = torch.sigmoid(torch.mm(x, matrix))

        x = (2 * np.pi) * x - np.pi
            
        return x  # (batch, 1)
    
    @property
    def description(self) -> str:
        """网络结构描述"""
        return (
            f"Network[分段输入={self.split_indices}, "
            f"全连接层={self.individual.gene_shapes[6:]}]"
        )

    def todev(self, device: str) -> nn.Module:
        """设备转移的增强实现"""
        self.device = torch.device(device)
        # 转移所有基因矩阵到新设备
        self.gene_matrices = [mat.to(device) for mat in self.gene_matrices]
        return self
    
if __name__=='__main__':
    # 创建个体并关联网络
    ind = Individual()
    net = NeuralNetwork(ind, device="cpu")

    # 批量前向计算 (支持任意batch_size)
    input_tensor = torch.ones(3, 19)  # 批量大小32
    output = net(input_tensor.to("cpu"))  # 形状 (32, 1)
    print(output)

    # 动态更新权重（当个体被遗传操作修改后）
    # ind.weights_tensor = new_weights_tensor
    # net.gene_matrices = net._parse_weights()  # 重新解析权重