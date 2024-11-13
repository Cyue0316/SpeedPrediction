import torch
import torch.nn as nn
import torch.optim as optim

# 自回归模型（AR Model）
class ARModel(nn.Module):
    def __init__(self, input_dim, output_dim, p=1):
        super(ARModel, self).__init__()
        self.p = p  # 自回归阶数
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 定义AR模型的权重
        self.weights = nn.Parameter(torch.randn(p, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        """
        x: input (B, T, N, C)
        output (B, T, N, 1)
        """
        B, T, N, C = x.shape
        # 初始化输出
        output = torch.zeros(B, T, N, 1).to(x.device)
        
        # 对每个节点进行自回归预测
        for t in range(self.p, T):
            for n in range(N):
                # 当前时间步的数据
                input_data = x[:, t-self.p:t, n, :]  # 取过去p个时间步的数据
                predicted_value = torch.sum(input_data * self.weights, dim=1) + self.bias  # 计算AR模型的输出
                output[:, t, n, 0] = predicted_value.squeeze()

        return output