import torch
import torch.nn as nn

class ARModel(nn.Module):
    def __init__(self, p, num_nodes):
        """
        p: 自回归阶数 表示AR模型使用过去p个时间步的数据进行预测
        num_nodes: 道路节点的数量
        """
        super(ARModel, self).__init__()

        self.p = p
        self.num_nodes = num_nodes
        
        # 定义共享的自回归参数和偏置
        self.ar_params = nn.Parameter(torch.randn(p))  # 形状为 (p,)
        self.bias = nn.Parameter(torch.zeros(1))  # 单个共享偏置

    def forward(self, x):
        """
        x: 输入数据，形状为 (B, T, N, 1)，其中 T=12
        返回预测值，形状为 (B, T, N, 1)
        """
        B, T, N, _ = x.size()
        
        # 检查时间步长度是否符合要求
        if T != self.p:
            raise ValueError(f"时间步 T ({T}) 必须等于自回归阶数 p ({self.p})。")

        # 构建输出张量，用于存储预测结果，确保它需要梯度
        output = torch.zeros(B, T, N, 1, device=x.device, dtype=x.dtype)

        # 批量计算预测值，使用 einsum 进行向量化操作
        prediction = torch.einsum('bpN,p->bN', x[:, :, :, 0], self.ar_params) + self.bias

        # 将预测值扩展为输出的形状 (B, T, N, 1)
        output = prediction.unsqueeze(1).unsqueeze(-1).expand(B, T, N, 1)
        
        return output
