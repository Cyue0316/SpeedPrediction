import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import numpy as np
from lib.data_prepare import get_edge_data_loader



class NConv(nn.Module):
    def __init__(self):
        super(NConv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class Linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(Linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(GCN,self).__init__()
        self.nconv = NConv()
        c_in = (order*support_len+1)*c_in
        self.mlp = Linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GWNet(nn.Module):
    def __init__(
        self,
        num_nodes,
        dropout,
        gcn_bool,
        addaptadj,
        in_dim,
        out_dim,
        nhid,
        kernel_size,
        blocks,
        layers,
        adj_path,
        adjinit=None):
        super(GWNet, self).__init__()


        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual_channels = nhid
        self.dilation_channels = nhid
        self.skip_channels = nhid*8
        self.end_channels = nhid*16
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.layers = layers
        self.adjinit = adjinit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.supports = self.load_adj_matrix(adj_path)
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.adjinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device), requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device), requires_grad=True).to(self.device)
                self.supports_len +=1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.adjinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1




        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1,self.kernel_size),dilation=new_dilation)) 

                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(self.dilation_channels,self.residual_channels,self.dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                  out_channels=self.end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    
    
    def load_adj_matrix(self, adj_path):
        num_nodes = self.num_nodes
        adj_matrix = get_edge_data_loader(adj_path) # 加载稀疏矩阵文件 (.npy)
        
        coo_matrix = adj_matrix.tocoo()
        # 提取COO格式矩阵的行和列
        # row = torch.tensor(coo_matrix.row, dtype=torch.long)
        # col = torch.tensor(coo_matrix.col, dtype=torch.long)
        # adj_tensor = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        
        # 获取COO格式的行和列索引
        row = coo_matrix.row
        col = coo_matrix.col
        
        # 将行列索引转换为 torch tensor，并移动到指定设备
        supports = [torch.tensor(row).to(self.device), torch.tensor(col).to(self.device)]
        
        return supports