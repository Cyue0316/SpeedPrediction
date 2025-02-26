import torch
import numpy as np
import os
import pandas as pd
import torch.utils
import torch.utils.data
from .utils import print_log, StandardScaler
import pydoop.hdfs as hdfs
import scipy.sparse as sp
import io
import datetime
from scipy.sparse import linalg
from sklearn.metrics.pairwise import cosine_similarity
from .slide_data import read_filled_data, reshape_data
# import geohash

# ! X shape: (B, T, N, C)


def read_hdfs_npy(file_path):
    with hdfs.open(file_path, 'rb') as f:
        data = np.load(f)
    return data



def get_dt_dataloaders(
    dt, mode='train',model='STAEFormer', batch_size=64, log=None
):
    file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
    data_x = read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
    data_y = read_hdfs_npy(file_path + "windowed_y.npy") # shape (288, 16326, 12, 1)
    data_x = data_x.transpose(0, 2, 1, 3)
    data_y = data_y.transpose(0, 2, 1, 3)
    # lens_df = pd.read_csv("/nfs/volume-65-1/lvyanming/gnn/SpeedPrediction/lib/link_properties.csv")
    # lens_values = lens_df['lens'].values
    # lens_values_expanded = lens_values[np.newaxis, np.newaxis, :]
    # data_x[..., 4] = lens_values_expanded / data_x[..., 4]
    # data_y[..., 0] = lens_values_expanded / data_y[..., 0]
    print(f"Data Shape:\tx-{data_x.shape}\ty-{data_y.shape}")
    if model == 'STAEFormer' or model == 'DCST':
        data_x = data_x[:, :, :400, [4, 1, 6, ]]
        data_y = data_y[:, :, :400, :]
        # 对 data_x[..., 1] 进行手动的归一化
        data_x_1 = data_x[..., 1]
        data_min = data_x_1.min()
        data_max = data_x_1.max()
        data_x_1_normalized = (data_x_1 - data_min) / (data_max - data_min)
        data_x[..., 1] = data_x_1_normalized

        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
    elif model == 'STID' or model == 'SAGE':
        data_x = data_x[:, :, :, [4, 1, 6]]
        data_y = data_y[:, :, :, :]
        # 对 data_x[..., 1] 进行手动的归一化
        # data_x_1 = data_x[..., 1]
        # data_min = data_x_1.min()
        # data_max = data_x_1.max()
        # data_x_1_normalized = (data_x_1 - data_min) / (data_max - data_min)
        # data_x[..., 1] = data_x_1_normalized
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        # batch_size = 32
    elif model == 'AR':
        data_x = data_x[:, :, :, [4]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        data_y = data_y[:, :, :, :]
    elif model == 'Attention':
        data_x = data_x[:, :, :, [4, 1, 6]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        data_y = data_y[:, :, :, :]
    elif model == 'AGCRN':
        data_x = data_x[:, :, :1000, [4]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        data_y = data_y[:, :, :1000, :]
        batch_size = 32
    elif model == 'GWNet':
        data_x = data_x[:, :, :6000, [4]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        data_y = data_y[:, :, :6000, :]
        batch_size = 32
        # data_x = data_x.transpose(0, 3, 2, 1)
        # data_y = data_y.transpose(0, 3, 2, 1)
    else:
        data_x = data_x[:, :, :, [4]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        data_y = data_y[:, :, :, :]

    print(f"Data Shape:\tx-{data_x.shape}\ty-{data_y.shape}")

    if mode == 'test':
        num_samples = data_x.shape[0]
        indices = torch.randperm(num_samples)
        shuffled_x = data_x[indices]
        shuffled_y = data_y[indices]
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(shuffled_x), torch.FloatTensor(shuffled_y)
        )
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
    else:
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(data_x), torch.FloatTensor(data_y)
        )
        if model == 'Attention':    # 计算采样参数
            num_batches = data_x.shape[2]  # N 维度数量
            nodes_per_batch = batch_size  # 每批次的节点数

            # 使用自定义的 NodeBatchSampler
            sampler = NodeBatchSampler(
                num_batches=num_batches, 
                nodes_per_batch=nodes_per_batch, 
                shuffle=True
            )

            dataset_loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=sampler  # 使用自定义的采样器
            )
        else:
            dataset_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
    print(f"Load data {dt} Finish!")
    if mode == 'test':
        return dataset_loader, scaler, indices
    return dataset_loader, scaler



class NodeBatchSampler(torch.utils.data.Sampler):
    def __init__(self, num_batches, nodes_per_batch, shuffle=True):
        """
        自定义采样器，用于按节点维度分批次采样。
        :param num_batches: 总共的批次数量 (B * ceil(N / nodes_per_batch))
        :param nodes_per_batch: 每个批次包含的节点数
        :param shuffle: 是否对节点顺序进行随机化
        """
        self.num_batches = num_batches
        self.nodes_per_batch = nodes_per_batch
        self.shuffle = shuffle

        # 生成节点索引 [0, N)
        self.node_indices = np.arange(self.num_batches * self.nodes_per_batch)
        if self.shuffle:
            np.random.shuffle(self.node_indices)

    def __iter__(self):
        """
        按批次生成采样索引。
        """
        for i in range(0, len(self.node_indices), self.nodes_per_batch):
            yield self.node_indices[i : i + self.nodes_per_batch]

    def __len__(self):
        """
        返回总批次数。
        """
        return len(self.node_indices) // self.nodes_per_batch
    
    

# not used
def get_val_dataloaders(
    dt_list, batch_size=64, log=None
):  
    # 用于存储所有日期的拼接数据
    all_data_x = []
    all_data_y = []

    for dt in dt_list:
        file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
        data_x = read_hdfs_npy(file_path + "windowed_data.npy")  # shape (288, 16326, 12, 8)
        data_y = read_hdfs_npy(file_path + "windowed_y.npy")  # shape (288, 16326, 12, 1)
        
        # 调整维度顺序
        data_x = data_x.transpose(0, 2, 1, 3)  # 转换为 (288, 12, 16326, 8)
        data_y = data_y.transpose(0, 2, 1, 3)    # 转换为 (288, 12, 16326, 1)
        data_x = data_x[:, :, :, [4, 1, 6]]  # 保留特定维度
        # 对数据进行归一化
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        
        data_x[..., 0] = scaler.transform(data_x[..., 0])

        # 将每个日期的 data_x 和 data_y 加入到列表中
        all_data_x.append(data_x)
        all_data_y.append(data_y)

        print_log(f"Trainset:\tx-{data_x.shape}\ty-{data_y.shape}", log=log)

    # 将所有日期的 data_x 和 data_y 沿着第一维度拼接
    all_data_x = np.concatenate(all_data_x, axis=0)  # 拼接后的形状是 (288 * len(dt_list), 12, 16326, 3)
    all_data_y = np.concatenate(all_data_y, axis=0)    # 拼接后的形状是 (288 * len(dt_list), 12, 16326, 1)

    # 创建 TensorDataset
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(all_data_x), torch.FloatTensor(all_data_y)
    )

    # 创建 DataLoader
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return dataset_loader, scaler



#------------------Graph Data Loader------------------#
def get_edge_data_loader(edge_path="hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/edge_adj.npy"):
    with hdfs.open(edge_path, 'rb') as f:
        adj = np.load(f)
    if adj.ndim == 2 and adj.shape[0] == adj.shape[1]:
        # 将邻接矩阵转为稀疏矩阵
        print("Kind Adj")
        # from scipy.sparse import coo_matrix
        # sparse_adj = coo_matrix(adj)
        # i = torch.LongTensor([sparse_adj.row, sparse_adj.col])  # row 和 col 分别为行和列的索引
        # v = torch.FloatTensor(sparse_adj.data)  # 稀疏矩阵的非零值
        # size = torch.Size(sparse_adj.shape)  # 稀疏矩阵的大小
        # # 创建稀疏张量
        # sparse_tensor = torch.sparse_coo_tensor(i, v, size)
        # adj_tensor = torch.tensor(adj, dtype=torch.float32)
    # 如果数据是边列表（每行两个节点）
    elif adj.ndim == 2 and adj.shape[0] == 2:
        print("Kind index")
        # edge_index = torch.tensor(adj, dtype=torch.long)  # 转换为 [2, E]
    else:
        raise ValueError("Invalid edge data format. Expected adjacency matrix or edge list.")
    
    return adj


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj, dtype=np.float32)  # 确保输入为稀疏矩阵且类型为浮点数
    rowsum = np.array(adj.sum(1)).flatten().astype(np.float32)  # 计算每行的度并转换为浮点数
    d_inv = np.zeros_like(rowsum, dtype=np.float32)  # 初始化倒数向量
    nonzero_mask = rowsum > 0  # 找出非零度的行
    d_inv[nonzero_mask] = np.power(rowsum[nonzero_mask], -1)  # 仅计算非零度的倒数
    d_mat = sp.diags(d_inv)  # 构造对角矩阵
    return d_mat.dot(adj).astype(np.float32).todense()  # 计算并返回标准化矩阵

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



#-------used in PatchSTG--------#


def seq2instance(data, P, Q):
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    # y = np.zeros(shape = (num_sample, P, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        # y[i] = data[i + Q : i + P + Q]
    return x

def read_meta():
    with hdfs.open("hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/link_properties.csv", 'rt') as f:
        meta = pd.read_csv(f)
        lat = meta['lat'].values
        lng = meta['lng'].values
        locations = np.stack([lat,lng], 0)
    return locations


def construct_adj(data):
    # construct the adj through the cosine similarity
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix-tem_matrix.mean())/tem_matrix.std())
    return tem_matrix



def augmentAlign(dist_matrix, auglen):
    # find the most similar points in other leaf nodes
    sorted_idx = np.argsort(dist_matrix.reshape(-1)*-1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)

# def reorderData(parts_idx, mxlen, adj, sps):
#     # parts_idx: segmented indices by kdtree
#     # adj: pad similar points through the cos_sim adj
#     # sps: spatial patch (small leaf nodes) size for padding
#     ori_parts_idx = np.array([], dtype=int)
#     reo_parts_idx = np.array([], dtype=int)
#     reo_all_idx = np.array([], dtype=int)
#     for i, part_idx in enumerate(parts_idx):
#         part_dist = adj[part_idx, :].copy()
#         part_dist[:, part_idx] = 0
#         if sps-part_idx.shape[0] > 0:
#             local_part_idx = augmentAlign(part_dist, sps-part_idx.shape[0])
#             auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
#         else:
#             auged_part_idx = part_idx

#         reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0])+sps*i])
#         ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
#         reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])
#     return ori_parts_idx, reo_parts_idx, reo_all_idx


def reorderData(parts_idx, mxlen, adj, sps, locations, dist_threshold = 0.011):
    """
    parts_idx: segmented indices by kdtree
    adj: pad similar points through the cos_sim adj
    sps: spatial patch (small leaf nodes) size for padding
    locations: [2, N] contains lng and lat of all nodes
    dist_threshold: distance threshold to determine if two regions are far apart
    """
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    region_marks = []  # To store region marks (1 or 0)
    
    # Calculate the center of each region (mean of coordinates)
    region_centers = []
    for part_idx in parts_idx:
        latitudes = locations[0, part_idx]  # Latitude
        longitudes = locations[1, part_idx]  # Longitude
        # Calculate the average coordinates for each region
        region_center = np.array([np.mean(latitudes), np.mean(longitudes)])
        region_centers.append(region_center)

    # Compare distance between adjacent regions
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if sps - part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        # Calculate distance between current region and the next region
        
        is_far = True  # Flag to check if the region should be marked as 1
        curr_region_center = region_centers[i]
        if i > 0:  # Check distance with previous region if it's not the first region
            prev_region_center = region_centers[i - 1]
            dist_to_prev = np.linalg.norm(curr_region_center - prev_region_center)
            if dist_to_prev <= dist_threshold:
                is_far = False
        
        if i < len(parts_idx) - 1:  # Check distance with next region if it's not the last region
            next_region_center = region_centers[i + 1]
            dist_to_next = np.linalg.norm(curr_region_center - next_region_center)
            if dist_to_next <= dist_threshold:
                is_far = False

        # Mark region as 1 if distance to both previous and next region are greater than threshold
        if is_far:
            region_marks.append(1)
        else:
            region_marks.append(0)
        
        # Collect reordered indices
        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps * i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])
    reduced_region_marks = []
    for i in range(0, len(region_marks), 8):
        chunk = region_marks[i:i+8]
        if len(chunk) < 8:  # In case the last group is smaller than 8 elements
            break
        # Count the number of 1's in the chunk
        if chunk.count(1) >= 4:  # If more than half are 1's, mark as 1
            reduced_region_marks.append(1)
        else:
            reduced_region_marks.append(0)
    return ori_parts_idx, reo_parts_idx, reo_all_idx, np.array(reduced_region_marks)

def kdTree(locations, times, axis):
    # locations: [2,N] contains lng and lat
    # times: depth of kdtree
    # axis: select lng or lat as hyperplane to split points
    sorted_idx = np.argsort(locations[axis])
    part1, part2 = np.sort(sorted_idx[:locations.shape[1]//2]), np.sort(sorted_idx[locations.shape[1]//2:])
    parts = []
    if times == 1:
        return [part1, part2], max(part1.shape[0], part2.shape[0])
    # if times == 1 or len(part1) <= 4 or len(part2) <= 4:
    #     return [part1, part2], max(len(part1), len(part2))
    else:
        left_parts, lmxlen = kdTree(locations[:,part1], times-1, axis^1)
        right_parts, rmxlen = kdTree(locations[:,part2], times-1, axis^1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    return parts, max(lmxlen, rmxlen)



def loadData(dt, mode='train', tod=288, dow=7):
    # Traffic
    file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
    data_x = read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
    data_y = read_hdfs_npy(file_path + "windowed_y.npy") # shape (288, 16326, 12, 1)
    data_x = data_x.transpose(0, 2, 1, 3)
    data_y = data_y.transpose(0, 2, 1, 3)
    data_x = data_x[:, :, :, [4, 1, 6, 3]] 
    scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
    data_x[..., 0] = scaler.transform(data_x[..., 0])
    print(f'Shape of data: {data_x.shape}')
    if mode == 'test':
        num_samples = data_x.shape[0]
        indices = torch.randperm(num_samples)
        shuffled_x = data_x[indices]
        shuffled_y = data_y[indices]
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(shuffled_x), torch.FloatTensor(shuffled_y)
        )
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False
        )
    else:
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(data_x), torch.FloatTensor(data_y)
        )
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True
        )
    print(f"Load data {dt} Finish!")
    if mode == 'test':
        return dataset_loader, scaler, indices
    return dataset_loader, scaler
    
def readStaticData(adjpath, dt="20240702", recurtimes=12, sps=4):
    locations = read_meta()
    print(f'Shape of locations: {locations.shape}')

    # load adj for padding
    if os.path.exists(adjpath):
        adj = np.load(adjpath)
    else:
        trainData = read_filled_data(dt)
        trainData = reshape_data(trainData, 288)
        trainData = trainData[...,4]
        trainData = trainData.transpose(1, 0)
        adj = construct_adj(trainData)
        np.save(adjpath, adj)
    # dt = (datetime.datetime.strptime(dt, '%Y%m%d') - datetime.timedelta(days=1)).strftime('%Y%m%d')
    # trainData = read_filled_data(dt)
    # trainData = reshape_data(trainData, 288)
    # trainData = trainData[...,4]
    # trainData = trainData.transpose(1, 0)
    # adj = construct_adj(trainData)
    # np.save(adjpath, adj)
    
    # partition and pad data with new indices
    parts_idx, mxlen = kdTree(locations, recurtimes, 0)
    ori_parts_idx, reo_parts_idx, reo_all_idx, area_index = reorderData(parts_idx, mxlen, adj, sps, locations)
    # print num of elements equal to 1 in area_index
    print(sum(1 for value in area_index if value == 1))
    print(len(area_index))
    # print(reo_all_idx.shape)
    # print(ori_parts_idx.shape)
    # print(ori_parts_idx == reo_parts_idx)
    # print(reo_all_idx.shape)
    # forward_reo_all_idx = np.roll(reo_all_idx, shift=4*sps)   # Forward offset by sps
    # backward_reo_all_idx = np.roll(reo_all_idx, shift=-4*sps)  # Backward offset by sps
    
    
    ##-- use directional attn
    # file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/20240701/'
    # data_x = read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
    # # data_x = data_x.transpose(0, 2, 1, 3)
    # data_x = data_x[0, :, 0, 0]
    
    
    ##-- if use GCN
    # adj = expand_adj_matrix(adj, reo_all_idx, sps)
    return ori_parts_idx, reo_parts_idx, reo_all_idx, area_index



# def geohashStatic(dt="20240702", recurtimes=12, sps=4):
#     locations = read_meta()
#     print(f'Shape of locations: {locations.shape}')
    
#     area_dict = dict()
#     for idx in range(locations.shape[1]):
#         id = geohash.encode(locations[0, idx], locations[1, idx], precision=6)
#         if id not in area_dict:
#             area_dict[id] = 0
#         area_dict[id] += 1
#     count_ones = sum(1 for value in area_dict.values() if value < 8)
#     print(len(area_dict.keys()))
#     print(max(area_dict.values()))
#     print(count_ones)
    # partition and pad data with new indices
    
    # parts_idx, mxlen = kdTree(locations, recurtimes, 0)
    # ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, mxlen, adj, sps)
    
    ##-- use directional attn
    # file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/20240701/'
    # data_x = read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
    # # data_x = data_x.transpose(0, 2, 1, 3)
    # data_x = data_x[0, :, 0, 0]
    
    
    ##-- if use GCN
    # adj = expand_adj_matrix(adj, reo_all_idx, sps)
    # return ori_parts_idx, reo_parts_idx, reo_all_idx
    

def geohashStatic(dt="20240702", recurtimes=12, sps=4):
    locations = read_meta()
    print(f'Shape of locations: {locations.shape}')
    
    area_dict = dict()
    for idx in range(locations.shape[1]):
        id = geohash.encode(locations[0, idx], locations[1, idx], precision=6)
        if id not in area_dict:
            area_dict[id] = 0
        area_dict[id] += 1

    print(f'Total areas: {len(area_dict)}')
    print(f'Max nodes in area: {max(area_dict.values())}')
    print(f'Areas with less than 8 nodes: {sum(1 for value in area_dict.values() if value < 8)}')
    print(f'Areas with more than 32 nodes: {sum(1 for value in area_dict.values() if value > 32)}')
    return area_dict

def merge_small_areas(area_dict):
    merged_dict = area_dict.copy()
    for area, count in list(area_dict.items()):
        if count < 8:
            # 获取相邻的区域，并尝试合并
            neighbors = geohash.neighbors(area)
            for neighbor in neighbors:
                if neighbor in merged_dict and merged_dict[neighbor] < 32:
                    merged_dict[neighbor] += count
                    del merged_dict[area]  # 删除当前区域
                    break
    return merged_dict

def split_large_area(area_dict, target_size=32):
    split_dict = {}
    for area, count in area_dict.items():
        if count > target_size:
            # 将大区域分割成小块
            sub_area_count = count // target_size
            remaining_count = count % target_size
            # 根据分割的数量划分区域（可以按经纬度分割或其他方式）
            for i in range(sub_area_count):
                split_dict[f'{area}_sub_{i}'] = target_size
            if remaining_count > 0:
                split_dict[f'{area}_sub_{sub_area_count}'] = remaining_count
        else:
            split_dict[area] = count
    return split_dict


def adjust_geohash_areas(target_nodes=32):
    area_dict = geohashStatic()  # 获取原始区域划分
    merged_area_dict = merge_small_areas(area_dict)  # 合并节点少的区域
    final_area_dict = split_large_area(merged_area_dict, target_size=target_nodes)  # 细分节点多的区域
    return final_area_dict



def output_area_stats(area_dict):
    # 输出总区域数
    total_areas = len(area_dict)
    # 获取每个区域的节点数量
    nodes_per_area = list(area_dict.values())
    
    # 计算区域内的节点数量的统计信息
    avg_nodes = np.mean(nodes_per_area)  # 平均节点数
    std_nodes = np.std(nodes_per_area)   # 节点数标准差
    max_nodes = np.max(nodes_per_area)   # 最大节点数
    min_nodes = np.min(nodes_per_area)   # 最小节点数

    # 输出区域划分统计信息
    print(f"Total number of areas: {total_areas}")
    print(f"Average number of nodes per area: {avg_nodes:.2f}")
    print(f"Standard deviation of nodes per area: {std_nodes:.2f}")
    print(f"Maximum nodes in an area: {max_nodes}")
    print(f"Minimum nodes in an area: {min_nodes}")
    
    # 可选：输出节点数量分布
    # print("\nDistribution of nodes per area:")
    # for area, node_count in area_dict.items():
    #     print(f"Area {area}: {node_count} nodes")


if __name__ == "__main__":
    # adj_path = "../data/dis_adj.npy"
    # readStaticData(adj_path)
    result_dict = adjust_geohash_areas()
    output_area_stats(result_dict)

