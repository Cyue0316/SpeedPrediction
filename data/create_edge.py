import numpy as np
import pydoop.hdfs as hdfs
import pandas as pd

def read_hdfs_npy(file_path):
    with hdfs.open(file_path, 'rb') as f:
        data = np.load(f)
    return data


def generate_edge_index(node_data, edge_text):
    # 1. 提取节点ID
    node_ids = node_data[0, 0, :, 0]  # 提取第一个batch第一个时间步的节点ID
    node_ids_str = [str(int(node_id)) for node_id in node_ids]
    # 2. 创建id到index的映射
    id2index = {node_id: idx for idx, node_id in enumerate(node_ids_str)}

    edge_index = []
    with hdfs.open(edge_text, 'rt') as f:
        for line in f:
            idA, idB = line.strip().split(' ')
            if idA in id2index and idB in id2index:
                indexA = id2index[idA]
                indexB = id2index[idB]
                edge_index.append((indexA, indexB))
                edge_index.append((indexB, indexA))  # 无向图添加反向边

    edge_index = np.array(edge_index).T  # 转置，使形状为 (num_edges, 2)
    return edge_index

def generate_adjacency_matrix(edge_index, num_nodes):
    # 创建标准邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    row = edge_index[0]
    col = edge_index[1]
    adj_matrix[row, col] = 1  # 在对应的位置上填充1，表示有边
    return adj_matrix



dt = "20240701"
file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
data_x = read_hdfs_npy(file_path + "windowed_data.npy")
data_x = data_x.transpose(0, 2, 1, 3)
print(data_x[0, 0, 0, 0])
edge_text_path = "hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/link_edge_data.txt"
edge_data = generate_edge_index(data_x, edge_text_path)
print("Edge data shape:", edge_data.shape)

# # 生成邻接矩阵
num_nodes = data_x.shape[2]  # 计算节点数
adj_matrix = generate_adjacency_matrix(edge_data, num_nodes)

# 输出邻接矩阵
print("Adjacency matrix shape:", adj_matrix.shape)
max_node_index = max(np.max(edge_data[0]), np.max(edge_data[1]))
print(f"Max node index: {max_node_index}")
print(f"Number of nodes: {num_nodes}")
# print(adj_matrix)
# np.save("edge_index.npy", edge_data)
# np.save("edge_adj.npy", adj_matrix)
