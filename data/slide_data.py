import numpy as np
import pydoop.hdfs as hdfs
import pandas as pd

def read_hdfs_file(file_path):
    with hdfs.open(file_path, 'rt') as f:
        lines = [line.strip().split(' ') for line in f]
    return lines

def reshape_data(df, time_slices):
    # 将数据按 linkid 和 tm 排序
    df = df.sort_values(by=['linkid', 'tm'])
    # 使用 groupby 将数据转为三维数组，同时保留 linkid
    reshaped = df.groupby('linkid').apply(
        lambda x: np.column_stack((x[['linkid']].values, x[['tm', 'car_cnt', 'status', 'speed', 'date', 'weekday', 'holiday']].values.reshape(time_slices, 7)))
    )
    return np.array(reshaped.tolist())


def read_filled_data(dt):
    raw_data_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/link_data_padding/{dt}/part-00000'
    filled_data_df = read_hdfs_file(raw_data_path)
    filled_data_df = pd.DataFrame(filled_data_df, columns=['linkid', 'tm', 'car_cnt', 'status', 'speed', 'date',
                                                             'weekday', 'holiday'])
    filled_data_df['tm'] = filled_data_df['tm'].astype(int)
    filled_data_df['car_cnt'] = filled_data_df['car_cnt'].astype(float).astype(int)
    filled_data_df['speed'] = filled_data_df['speed'].astype(float)
    filled_data_df['date'] = filled_data_df['date'].str[-4:]
    return filled_data_df

if __name__ == '__main__':
    # 读取填充后的数据
    dt = "20240701"
    filled_data_df = read_filled_data(dt)
    # 读取前一天的数据
    prev_dt = "20240630"  # 前一天的日期
    prev_filled_data_df = read_filled_data(prev_dt)
    # 读取后一天的数据
    next_dt = "20240702"  # 后一天的日期
    next_filled_data_df = read_filled_data(next_dt)
    
    # 获取 prev_filled_data_df 的最后 12 个时间片
    prev_last_12 = prev_filled_data_df.groupby('linkid').tail(12)
    # 获取 next_filled_data_df 的前 11 个时间片
    next_first_11 = next_filled_data_df.groupby('linkid').head(11)

    # 确保三个 DataFrame 按 linkid 对齐，并找到交集
    common_linkids = set(prev_last_12['linkid']).intersection(set(filled_data_df['linkid']))\
        .intersection(set(next_first_11['linkid']))
    prev_last_12_filtered = prev_last_12[prev_last_12['linkid'].isin(common_linkids)]
    filled_data_filtered = filled_data_df[filled_data_df['linkid'].isin(common_linkids)]
    next_first_11_filtered = next_first_11[next_first_11['linkid'].isin(common_linkids)]

    prev_last_12_array = reshape_data(prev_last_12_filtered, 12)
    filled_data_array = reshape_data(filled_data_filtered, 288)
    next_first_11_array = reshape_data(next_first_11_filtered, 11)

    # 按时间片维度进行拼接
    combined_array = np.concatenate((prev_last_12_array, filled_data_array, next_first_11_array), axis=1)

    # 输出结果
    print(combined_array.shape)
    print(combined_array[0,11:13,:])
    print(combined_array[0,299:301,:])
    
    # 定义滑动窗口的大小
    window_size = 12
    time_slots = combined_array.shape[1] - window_size * 2 + 1

    # 创建滑动窗口数组
    windowed_data = np.empty((combined_array.shape[0], time_slots, window_size, combined_array.shape[2]))
    windowed_y = np.empty((combined_array.shape[0], time_slots, window_size, 1))

    # 进行滑动窗口处理
    for i in range(time_slots):
        windowed_data[:, i, :, :] = combined_array[:, i:i + window_size, :]
        windowed_y[:, i, :, 0] = combined_array[:, i + window_size:i + window_size * 2, 4]  # 取第5个特征speed

    # 输出windowed_data和windowed_y的shape
    print("windowed_data")
    print(windowed_data.shape)
    print("windowed_y")
    print(windowed_y.shape)
    
    # print(windowed_y[0, 0, :, 0])
    np.save('windowed_data.npy', windowed_data)
    np.save('windowed_y.npy', windowed_y) 
