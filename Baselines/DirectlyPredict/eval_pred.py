import lib.metrics as metrics
import lib.data_prepare as data_prepare
import numpy as np
import datetime


dt = "20240630"
dt_list = [(datetime.datetime.strptime(dt, '%Y%m%d') + datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range(110)]
total_len = len(dt_list)
train_len = int(total_len * 0.6)
val_len = int(total_len * 0.2)

# 划分
train_list = dt_list[:train_len]
val_list = dt_list[train_len:]
with open("train_eval.txt", "w") as f:
    for now_dt in dt_list:
        file_path = f'hdfs://DClusterNmg3:8020/user/bigdata-dp/lvyanming/traffic_map/exp_data/sliding_data/{now_dt}/'
        data_x = data_prepare.read_hdfs_npy(file_path + "windowed_data.npy") # shape (288, 16326, 12, 8)
        data_y = data_prepare.read_hdfs_npy(file_path + "windowed_y.npy") # shape (288, 16326, 12, 1)
        data_x = data_x.transpose(0, 2, 1, 3)
        data_y = data_y.transpose(0, 2, 1, 3)
        data_x = data_x[:, :, :, [4, 1, 6]]
        data_y = data_y[:, :, :, :]
        y_pred = data_x[..., 0]
        data_y = np.squeeze(data_y, axis=-1)
        # print(now_dt)
        rmse, mae, mape = metrics.RMSE_MAE_MAPE(data_y, y_pred)
        f.write(f"{now_dt}\t{rmse}\t{mae}\t{mape}\n")
