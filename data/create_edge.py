import pydoop.hdfs as hdfs
import numpy as np
import pandas as pd


def read_hdfs_file(file_path):
    with hdfs.open(file_path, 'rt') as f:
        lines = [line.strip().split(' ') for line in f]
    return lines

# edge_path = "hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/link_edge/part-00000"
# edge_df = read_hdfs_file(edge_path)
# print(len(edge_df))
# link_path = "hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/final_link_range/part-00000"
# link_list = read_hdfs_file(link_path)

# # 将link_list展平为集合
# link_set = set([link[0] for link in link_list])

# # 过滤edge_df，仅保留两列都在link_set中的行
# filtered_edge_df = [edge for edge in edge_df if edge[0] in link_set and edge[1] in link_set]
# filtered_edge_df.sort(key=lambda x: (x[0], x[1]))
# print(len(filtered_edge_df))
# # 将结果保存为文件
# output_path = "hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/link_edge_data.txt"
# with hdfs.open(output_path, 'wt') as f:
#     for edge in filtered_edge_df:
#         f.write(' '.join(edge) + '\n')
        

def build_node_mapping_from_x(x):

    # 提取所有节点ID
    node_ids = x[0, 0, :, 0].tolist()  # 从第一个batch和时间步提取节点ID，假设节点ID在所有batch中一致
    node_mapping = {node_id: idx for idx, node_id in enumerate(node_ids)}
    return node_mapping

def read_hdfs_npy(file_path):
    with hdfs.open(file_path, 'rb') as f:
        data = np.load(f)
    return data


def build_edge_index(edge_data, node_mapping):
    """
    根据节点映射关系将边数据转换为 NumPy 格式的 edge_index。
    
    参数:
    - edge_data: 包含边的DataFrame 包含 'nodeid1' 和 'nodeid2' 列
    - node_mapping: dict 节点ID到整数索引的映射

    返回:
    - edge_index: np.ndarray 形状为(2, num_edges)
    """
    edge_index = np.array(
        [(node_mapping[nodeid1], node_mapping[nodeid2]) for nodeid1, nodeid2 in zip(edge_data['nodeid1'], edge_data['nodeid2'])],
        dtype=np.int64
    ).T
    return edge_index


dt='20240701'
file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{dt}/'
data_x = read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
edge_path = "hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/link_edge/part-00000"
edge_df = read_hdfs_file(edge_path)

# 构建节点映射并生成 edge_index
node_mapping = build_node_mapping_from_x(data_x)
edge_index = build_edge_index(edge_data, node_mapping)

print("Node Mapping:", node_mapping)
print("Edge Index:\n", edge_index)

