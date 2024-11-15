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
        
        # 为每个节点定义一个独立的自回归参数
        self.ar_params = nn.Parameter(torch.randn(num_nodes, p))  # 形状为 (N, p)
        self.bias = nn.Parameter(torch.zeros(num_nodes))  # 每个节点的偏置

    def forward(self, x):
        """
        x: 输入数据，形状为 (B, T, N, 1)
        返回预测值，形状为 (B, T, N, 1)
        """
        B, T, N, _ = x.size()
        
        # 输出张量，用于存储预测结果，形状为 (B, T, N, 1)
        output = torch.zeros(B, T, N, 1).to(x.device)
        
        # 对每个节点进行自回归预测
        for b in range(B):
            for n in range(N):
                # 取出当前节点的历史数据，形状为 (T,)
                history = x[b, :, n, 0].clone()  # 使用clone()防止原地操作，shape: (T,)
                
                # 预测未来的每个时间步
                for t in range(T):
                    # 使用自回归参数和历史数据进行预测
                    prediction = torch.matmul(self.ar_params[n], history[-self.p:]) + self.bias[n]
                    
                    # 将预测值赋给输出张量
                    output[b, t, n, 0] = prediction
                    
                    # 更新历史数据，移除最早的时间步，加入新预测的值
                    history = torch.cat([history[1:], prediction.unsqueeze(0)], dim=0)
        
        return output
