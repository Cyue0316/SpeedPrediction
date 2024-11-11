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


# def get_dataloaders_from_index_data(
#     dt, batch_size=64, log=None
# ):
#     # dt = "20240630"
#     file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
#     data_x = read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
#     data_y = read_hdfs_npy(file_path + "windowed_y.npy") # shape (288, 16326, 12, 1)
#     # data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

#     # features = [0]
#     # if tod:
#     #     features.append(1)
#     # if dow:
#     #     features.append(2)
#     # # if dom:
#     # #     features.append(3)
#     # data = data[..., features]

#     # index = np.load(os.path.join(data_dir, "index.npz"))
#     # train_index = index["train"]  # (num_samples, 3)
#     # val_index = index["val"]
#     # test_index = index["test"]

#     x_train_index = vrange(train_index[:, 0], train_index[:, 1])
#     y_train_index = vrange(train_index[:, 1], train_index[:, 2])
#     x_val_index = vrange(val_index[:, 0], val_index[:, 1])
#     y_val_index = vrange(val_index[:, 1], val_index[:, 2])
#     x_test_index = vrange(test_index[:, 0], test_index[:, 1])
#     y_test_index = vrange(test_index[:, 1], test_index[:, 2])

#     x_train = data[x_train_index]
#     y_train = data[y_train_index][..., :1]
#     x_val = data[x_val_index]
#     y_val = data[y_val_index][..., :1]
#     x_test = data[x_test_index]
#     y_test = data[y_test_index][..., :1]

#     scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

#     x_train[..., 0] = scaler.transform(x_train[..., 0])
#     x_val[..., 0] = scaler.transform(x_val[..., 0])
#     x_test[..., 0] = scaler.transform(x_test[..., 0])

#     print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
#     print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
#     print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

#     trainset = torch.utils.data.TensorDataset(
#         torch.FloatTensor(x_train), torch.FloatTensor(y_train)
#     )
#     valset = torch.utils.data.TensorDataset(
#         torch.FloatTensor(x_val), torch.FloatTensor(y_val)
#     )
#     testset = torch.utils.data.TensorDataset(
#         torch.FloatTensor(x_test), torch.FloatTensor(y_test)
#     )

#     trainset_loader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True
#     )
#     valset_loader = torch.utils.data.DataLoader(
#         valset, batch_size=batch_size, shuffle=False
#     )
#     testset_loader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False
#     )

#     return trainset_loader, valset_loader, testset_loader, scaler


def get_dt_dataloaders(
    dt, batch_size=64, log=None
):
    file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
    data_x = read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
    datay = read_hdfs_npy(file_path + "windowed_y.npy") # shape (288, 16326, 12, 1)
    data_x = data_x.transpose(0, 2, 1, 3)
    datay = datay.transpose(0, 2, 1, 3)
    data_x = data_x[:, :, :250, [4, 1, 6]]
    datay = datay[:, :, :250, :]

    scaler = StandardScaler(mean=data_x[..., 0].mean(), std=data_x[..., 0].std())
    data_x[..., 0] = scaler.transform(data_x[..., 0])
    # 获取 data_x[..., 1] 部分
    data_x_1 = data_x[..., 1]

    # 计算 min 和 max 值
    data_min = data_x_1.min()
    data_max = data_x_1.max()


    # 对 data_x[..., 1] 进行手动的归一化
    data_x_1_normalized = (data_x_1 - data_min) / (data_max - data_min)

    # 将标准化后的 data_x[..., 1] 重新赋值回 data_x
    data_x[..., 1] = data_x_1_normalized

    print_log(f"Data Shape:\tx-{data_x.shape}\ty-{datay.shape}", log=log)

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

