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

    return np.array(edge_index).T



dt = "20240701"
file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
data_x = read_hdfs_npy(file_path + "windowed_data.npy")
data_x = data_x.transpose(0, 2, 1, 3)
print(data_x[0, 0, 0, 0])
edge_text_path = "hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/link_edge_data.txt"
edge_data = generate_edge_index(data_x, edge_text_path)
print(edge_data.shape)

# data_x = data_x.transpose(0, 2, 1, 3)
# data_x = data_x[0, 0, :400, 0]
# data_x = data_x.astype(int)






# node_pairs = [
#     (152, 252), (153, 253), (20, 120), (225, 325), (191, 291), 
#     (164, 264), (161, 261), (220, 320), (19, 119), (20, 220), 
#     (224, 324), (44, 144), (189, 289), (61, 161), (120, 220), 
#     (158, 258), (197, 297), (97, 197), (156, 256), (190, 290)
# ]

# 输出节点对应的名称
# for rank, (node1, node2) in enumerate(node_pairs, start=1):
#     name1 = data_x[node1]
#     name2 = data_x[node2]
#     print(f"Rank {rank}: {name1},{name2}")

# def save_to_txt_one_line(data, output_file):
#     # 将数据转换为逗号分隔的字符串格式并保存到文件中
#     with open(output_file, 'w') as f:
#         f.write(",".join(map(str, data)))


# # 输出文件路径
# output_file_path = "data_output.txt"
# save_to_txt_one_line(data_x, output_file_path)
