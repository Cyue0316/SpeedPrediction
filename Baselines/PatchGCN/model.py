import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # split into query, key, value

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query、Key、Value 的线性映射
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        """
        query: 来自当前区域或当前任务的特征 (B, N_query, C)
        key_value: 来自其他区域或背景任务的特征 (B, N_key_value, C)
        """
        B, N_query, C = query.shape
        B2, N_key_value, C2 = key_value.shape
        assert B == B2, "Batch size of query and key_value must match."
        assert C == C2, "The dimension of query and key_value must be the same."

        # Query、Key、Value 的线性变换
        kv = self.kv(key_value) \
            .reshape(B, N_key_value, 2, self.num_heads, self.head_dim) \
            .permute(2, 0, 3, 1, 4)
        q = self.q(query) \
            .reshape(B, N_query, self.num_heads, self.head_dim) \
            .permute(0, 2, 1, 3)

        q, k, v = q, kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_query, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    

# class GCNLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_channels, out_channels)
        
#     def forward(self, x, adj):
#         """
#         x: node features (B, N, D)
#         adj: adjacency matrix (B, N, N) for each graph
#         """
#         # GCN layer implementation: x' = A * x * W
#         # x = torch.matmul(adj, x)  # A * x
#         batch_size, num_blocks, block_size, feature_dim = x.shape
#         # adj = torch.bmm(nv1, nv1.transpose(1, 2))
#         # adj = F.softmax(F.relu(adj), dim=-1)
#         assert num_blocks == adj.shape[0], "邻接矩阵块数应与输入块数一致"
#         assert block_size == adj.shape[1], "邻接矩阵大小应与块大小一致"
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, num_blocks, block_size, self.linear.out_features, device=x.device)
        
#         # 遍历每个小块，应用邻接矩阵
#         for i in range(num_blocks):
#             # 获取当前块的邻接矩阵 (block_size, block_size)
#             current_adj = adj[i]
#             # 获取当前块的输入数据 (batch_size, block_size, feature_dim)
#             current_input = x[:, i, :, :]
#             # 计算 A * X
#             current_output = torch.einsum('nd,bdh->bnh', current_adj, current_input)
#             # 线性变换 (W)
#             current_output = self.linear(current_output)
#             # 存储结果
#             output[:, i, :, :] = current_output
#         # print(output.shape)
#         return output


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, adj):
        """
        x: node features (B, N, S, D_in)
        adj: adjacency matrix (N, S, S) for each block
        """
        # 批量矩阵乘法优化
        adj_expanded = adj.unsqueeze(0)  # 扩展维度 (1, N, S, S)
        x_transformed = torch.matmul(adj_expanded, x)  # (B, N, S, D_in)
        
        # 应用线性变换
        return self.linear(x_transformed)
    
        

class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size,mlp_ratio=4.0, cross=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size
        self.cross = cross

        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.gcn = GCNLayer(hidden_size, hidden_size)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.cattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        # self.bcattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        # self.sgcn = GCNLayer(hidden_size, hidden_size)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x, adj, area_index):
        B,T,_,D = x.shape

        # P: patch num and N: patch size
        P, N = self.num, self.size
        assert self.num * self.size == x.shape[2]
        # (batch_size, 12, 512, 32, 160)
        x = x.reshape(B, T, P, N, D)

        # depth attention
        if self.cross:
            cross_mask = torch.tensor(area_index).to(x.device) == 1
            self_mask = ~cross_mask
            
            x_forward = torch.roll(x, shifts=1, dims=2)
            x_backward = torch.roll(x, shifts=-1, dims=2)
            q = self.snorm1(x.reshape(B*T*P,N,D))
            kv = torch.cat([x_forward, x_backward], dim=3)
            kv = self.snorm1(kv.reshape(B*T*P,2*N,D))
            if cross_mask.any():
                cross_attn_out = self.cattn(q, kv).reshape(B, T, P, N, D)
                x = x + cross_attn_out * cross_mask.view(1, 1, P, 1, 1)
            if self_mask.any():
                self_attn_out = self.sattn(q).reshape(B, T, P, N, D)
                x = x + self_attn_out * self_mask.view(1, 1, P, 1, 1)
                # x = x + self.sattn(q).reshape(B,T,P,N,D)
            # x = x + self.bcattn(q, backward_kv).reshape(B,T,P,N,D)
            
        else:
            # qkv = self.snorm1(x.reshape(B*T,P,N,D))
            qkv = self.snorm1(x.reshape(B*T*P,N,D))
            # x = x + self.sgcn(qkv, sadj).reshape(B,T,P,N,D)
            x = x + self.sattn(qkv).reshape(B,T,P,N,D)
        x = x + self.smlp(self.snorm2(x))
        
        # breadth attention
        # qkv = self.nnorm1(x.transpose(2,3).reshape(B*T*N,P,D))
        qkv = self.nnorm1(x.transpose(2,3).reshape(B*T,N,P,D))
        # print(qkv.shape)
        x = x + self.gcn(qkv, adj).reshape(B,T,N,P,D).transpose(2, 3)
        x = x + self.nmlp(self.nnorm2(x))
        return x.reshape(B,T,-1,D)
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: (B * N, D, T)
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)  # (B * N, D, T)
        return x
    
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # x shape: (B, T, N, D)
        B, T, N, _ = x.shape
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # (B, N, T)
        x = x.reshape(B * N, 1, T)  # (B * N, 1, T)

        # 计算移动平均值
        moving_mean = self.moving_avg(x)  # (B * N, 1, T)

        # 计算残差
        res = x - moving_mean  # (B * N, 1, T)

        # 恢复形状
        moving_mean = moving_mean.reshape(B, N, T).permute(0, 2, 1)  # (B, T, N)
        res = res.reshape(B, N, T).permute(0, 2, 1).unsqueeze(-1)   # (B, T, N)

        # 将结果合并回原始数据
        moving_mean = moving_mean.unsqueeze(-1)  # (B, T, N, 1)
        
        return moving_mean, res

class PatchGCN(nn.Module):
    def __init__(self, tem_patchsize, tem_patchnum,
                        node_num, spa_patchsize, spa_patchnum,
                        tod, dow,
                        layers, factors,
                        input_dims, node_dims, tod_dims, dow_dims
                ):
        super(PatchGCN, self).__init__()
        self.node_num = node_num

        self.tod, self.dow = tod, dow

        status_dims = 8
        # model_dims = input_emb + spa_emb + tem_emb + status_emb
        dims = input_dims + tod_dims + dow_dims + node_dims + status_dims

        # spatio-temporal embedding -> section 4.1 in paper
        
        ##-- avg_smooth
        self.serise_decomp = series_decomp(kernel_size=tem_patchsize+1)
        
        # input_emb
        self.input_st_fc = nn.Conv2d(in_channels=4, out_channels=input_dims, kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)
        # spa_emb
        self.node_emb = nn.Parameter(
                torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        # tem_emb
        self.time_in_day_emb = nn.Parameter(
                torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        ##-- status_emb
        self.status_emb = nn.Parameter(
                torch.empty(4, status_dims))
        nn.init.xavier_uniform_(self.status_emb)

        # dual attention encoder -> section 4.3 in paper, factors for merging the leaf nodes of KDTree
        self.spa_encoder = nn.ModuleList([])
        for i in range(layers):
            if i == 1 or i == 3:
                self.spa_encoder.append(WindowAttBlock(dims, 1, spa_patchnum//factors, spa_patchsize*factors, mlp_ratio=1, cross=True))
            else:
                self.spa_encoder.append(WindowAttBlock(dims, 1, spa_patchnum//factors, spa_patchsize*factors, mlp_ratio=1))
        # projection decoder -> section 4.4 in paper
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum*dims, out_channels=tem_patchsize*tem_patchnum, kernel_size=(1, 1), bias=True)
        
        ##-- adj init
        # self.nv1 = nn.Parameter(torch.empty(spa_patchsize * factors, spa_patchnum // factors, 64))
        # self.nv2 = nn.Parameter(torch.empty(spa_patchsize * factors, 64, spa_patchnum // factors))
        self.adj = nn.Parameter(torch.empty(spa_patchsize*factors, spa_patchnum//factors, spa_patchnum//factors))
        nn.init.xavier_uniform_(self.adj)
        
        ##-- region init
        # self.region_emb = nn.Parameter(torch.empty(spa_patchnum//factors, 32))
        # nn.init.xavier_uniform_(self.region_emb)
        
        # self.sadj = nn.Parameter(torch.empty(spa_patchnum//factors, spa_patchsize*factors, spa_patchsize*factors))
        # nn.init.xavier_uniform_(self.sadj)
        # self.nv1 = nn.Parameter(torch.empty(spa_patchsize * factors, spa_patchnum // factors, 32))
        # nn.init.xavier_uniform_(self.nv1)

    
    def set_index(self, ori_parts_idx, reo_parts_idx, reo_all_idx, area_index=None):
        self.ori_parts_idx = ori_parts_idx
        self.reo_parts_idx = reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.area_index = area_index
        # self.adj = adj
    
    
    
    def forward(self, x):
        # spatio-temporal embedding -> section 4.1 in paper
        embeded_x = self.embedding(x)
        rex = embeded_x[:,:,self.reo_all_idx,:] # select patched points
        # B, T, N, _ = rex.shape
        # P, _ = self.region_emb.shape
        # region_emb = self.region_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # region_emb = region_emb.expand(B, T, N // P, P, -1).transpose(2, 3)

        # region_emb = region_emb.reshape(B,T,N,-1)

        # rex = torch.cat([rex, region_emb], dim=-1)

        # dual attention encoder -> section 4.3 in paper
        # self.adj = torch.tensor(self.adj, dtype=torch.float32).to(x.device)
        for block in self.spa_encoder:
            # rex = block(rex)
            # rex = block(rex, self.adj, self.sadj)
            rex = block(rex, self.adj, self.area_index)

        orginal = torch.zeros(rex.shape[0],rex.shape[1],self.node_num,rex.shape[-1]).to(x.device)
        orginal[:,:,self.ori_parts_idx,:] = rex[:,:,self.reo_parts_idx,:] # back to the original indices

        # projection decoder -> section 4.4 in paper
        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y # [B,T,N,1]
    
    

    

    def embedding(self, x):
        b,t,n,_ = x.shape
        # x1: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information

        # te = x[..., 1:] # [B,T,N,2]
        # x1 = x[..., :1]
        # input traffic + time of day + day of week as the input signal
        x_smooth, _ = self.serise_decomp(x[..., :1])
        x1 = torch.cat([
            x_smooth,  # Traffic
            (x[..., 1:2] / self.tod),  # Time of day normalized
            (x[..., 2:3] / self.dow),  # Day of week normalized
            (x[..., 3:4] / 4) # status data normalized
        ], -1).float()
        # Process input data
        input_data = self.input_st_fc(x1.transpose(1, 3)).transpose(1, 3)
        t, d = input_data.shape[1], input_data.shape[-1]

        # Append time of day embedding
        t_i_d_data = x[:, -t:, :, 1]
        input_data = torch.cat([input_data, self.time_in_day_emb[t_i_d_data.long()]], -1)

        # Append day of week embedding
        d_i_w_data = x[:, -t:, :, 2]
        input_data = torch.cat([input_data, self.day_in_week_emb[d_i_w_data.long()]], -1)
        
        ##-- Append status embedding
        status_data = x[:, -t:, :, 3]
        status_data = status_data - 1
        input_data = torch.cat([input_data, self.status_emb[status_data.long()]], -1)

        # Append spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        # print(node_emb.shape)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data