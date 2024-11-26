import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
import pydoop.hdfs as hdfs

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
        data_x = data_x[:, :, :5000, [4, 1, 6, 2]]
        scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
        data_x[..., 0] = scaler.transform(data_x[..., 0])
        datay = datay[:, :, :5000, :]
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

