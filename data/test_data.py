import numpy as np
import pydoop.hdfs as hdfs
import pandas as pd

def read_hdfs_npy(file_path):
    with hdfs.open(file_path, 'rb') as f:
        data = np.load(f)
    return data

dt = "20240701"
file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
data_x = read_hdfs_npy(file_path + "windowed_data.npy")
data_x = data_x.transpose(0, 2, 1, 3)
data_x = data_x[:, :, :250, [4, 1, 6]]
# print(data_x.shape)
print(data_x[0, 0, 0, :])
# 筛选第二维度等于288的data_x
data_x = data_x[data_x[:, :, :, 1] == 288]
print(data_x.shape)
print(data_x)
# print(data_x[0, 0, 0, :])