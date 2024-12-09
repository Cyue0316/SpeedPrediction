import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
import pydoop.hdfs as hdfs
import scipy.sparse as sp
import io
from scipy.sparse import linalg
from scipy.sparse import coo_matrix

# ! X shape: (B, T, N, C)


def read_hdfs_npy(file_path):
    with hdfs.open(file_path, 'rb') as f:
        data = np.load(f)
    return data



def get_dt_dataloaders(
    dt, model='STAEFormer', batch_size=64, log=None
):
    file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
    data_x = read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
    datay = read_hdfs_npy(file_path + "windowed_y.npy") # shape (288, 16326, 12, 1)
    data_x = data_x.transpose(0, 2, 1, 3)
    datay = datay.transpose(0, 2, 1, 3)
    # print(f"Data Shape:\tx-{data_x.shape}\ty-{datay.shape}")
    
    if model == 'STAEFormer' or model == 'DCST':
        data_x = data_x[:, :, :400, [4, 1, 6, ]]
        datay = datay[:, :, :400, :]
        

        # 对 data_x[..., 1] 进行手动的归一化
        data_x_1 = data_x[..., 1]
        data_min = data_x_1.min()
        data_max = data_x_1.max()
        data_x_1_normalized = (data_x_1 - data_min) / (data_max - data_min)
        data_x[..., 1] = data_x_1_normalized

        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
    elif model == 'AR':
        data_x = data_x[:, :, :, [4]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        datay = datay[:, :, :, :]
    elif model == 'temp':
        data_x = data_x[:, :, :16000, [4, 1, 6, 2]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        datay = datay[:, :, :16000, :]
    elif model == 'GWNet':
        data_x = data_x[:, :, :6000, [4]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        datay = datay[:, :, :6000, :]
        batch_size = 32
        # data_x = data_x.transpose(0, 3, 2, 1)
        # datay = datay.transpose(0, 3, 2, 1)
    else:
        data_x = data_x[:, :, :10000, [4]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        datay = datay[:, :, :10000, :]

    print(f"Data Shape:\tx-{data_x.shape}\ty-{datay.shape}")
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data_x), torch.FloatTensor(datay)
    )

    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    print(f"Load data {dt} Finish!")
    return dataset_loader, scaler

# not used
def get_val_dataloaders(
    dt_list, batch_size=64, log=None
):  
    # 用于存储所有日期的拼接数据
    all_data_x = []
    all_datay = []

    for dt in dt_list:
        file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
        data_x = read_hdfs_npy(file_path + "windowed_data.npy")  # shape (288, 16326, 12, 8)
        datay = read_hdfs_npy(file_path + "windowed_y.npy")  # shape (288, 16326, 12, 1)
        
        # 调整维度顺序
        data_x = data_x.transpose(0, 2, 1, 3)  # 转换为 (288, 12, 16326, 8)
        datay = datay.transpose(0, 2, 1, 3)    # 转换为 (288, 12, 16326, 1)
        data_x = data_x[:, :, :, [4, 1, 6]]  # 保留特定维度
        # 对数据进行归一化
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        
        data_x[..., 0] = scaler.transform(data_x[..., 0])

        # 将每个日期的 data_x 和 datay 加入到列表中
        all_data_x.append(data_x)
        all_datay.append(datay)

        print_log(f"Trainset:\tx-{data_x.shape}\ty-{datay.shape}", log=log)

    # 将所有日期的 data_x 和 datay 沿着第一维度拼接
    all_data_x = np.concatenate(all_data_x, axis=0)  # 拼接后的形状是 (288 * len(dt_list), 12, 16326, 3)
    all_datay = np.concatenate(all_datay, axis=0)    # 拼接后的形状是 (288 * len(dt_list), 12, 16326, 1)

    # 创建 TensorDataset
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(all_data_x), torch.FloatTensor(all_datay)
    )

    # 创建 DataLoader
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return dataset_loader, scaler



#------------------Graph Data Loader------------------#
def get_edge_data_loader(edge_path):
    with hdfs.open(edge_path, 'rb') as f:
        adj = np.load(f)
    edge_index = torch.tensor(adj, dtype=torch.long)
    print(edge_index.shape)
    # adj = adj[:500, :500]
    # adj = [asym_adj(adj), asym_adj(np.transpose(adj))]
    # adj_matrix = asym_adj(adj)
    return edge_index


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

