import torch
from torch import nn
import torch.functional as F
from lib.data_prepare import get_edge_data_loader

class GCNLayer(nn.Module):
    """带残差连接的 Graph Convolutional Network (GCN) 层"""

    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): 输入特征的维度。
            output_dim (int): 输出特征的维度。
        """
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)  # 图卷积的线性变换
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)

        # 如果输入输出维度不一致，添加一个映射层
        self.residual = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()

    def forward(self, x, edge_index):
        """
        Args:
            x (torch.Tensor): 节点特征矩阵，形状为 [B, D, N, 1]。
            edge_index (torch.Tensor): 图的邻接矩阵，形状为 [N, N]。

        Returns:
            torch.Tensor: 更新后的节点特征，形状为 [B, D_out, N, 1]。
        """
        B, D, N, _ = x.shape
        adj = edge_index  # 邻接矩阵

        # 对邻接矩阵进行归一化：D^{-1/2} * A * D^{-1/2}
        I = torch.eye(N, device=x.device)  # 单位矩阵
        adj = adj + I  # 加上自环
        D = torch.diag(torch.pow(adj.sum(dim=1), -0.5))  # 计算度矩阵 D^{-1/2}
        adj = torch.mm(torch.mm(D, adj), D)  # 归一化邻接矩阵

        # GCN 卷积操作
        x = x.squeeze(-1).permute(0, 2, 1)  # 形状变为 [B, N, D]
        out = torch.matmul(adj, self.linear(x))  # 图卷积
        out = out.permute(0, 2, 1).unsqueeze(-1)  # 恢复形状为 [B, D_out, N, 1]

        # 残差连接
        res = self.residual(x.permute(0, 2, 1).unsqueeze(-1))  # [B, D_out, N, 1]
        out = self.dropout(self.act(out + res))  # 激活函数 + Dropout
        return out

    
    
class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, num_nodes, node_dim, input_len, input_dim, embed_dim, output_len, num_layer, if_node, if_T_i_D, if_D_i_W, temp_dim_tid, temp_dim_diw, time_of_day_size, day_of_week_size):
        super().__init__()
        # attributes
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.input_len = input_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.num_layer = num_layer
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        self.if_time_in_day = if_T_i_D
        self.if_day_in_week = if_D_i_W
        self.if_spatial = if_node
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_index = get_edge_data_loader()

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[GCNLayer(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            # time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            # day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        print(hidden.shape)
        # encoding
        edge_index = torch.tensor(self.edge_index, dtype=torch.float32).to(self.device)  # 邻接矩阵
        for layer in self.encoder:
            hidden = layer(hidden, edge_index)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction