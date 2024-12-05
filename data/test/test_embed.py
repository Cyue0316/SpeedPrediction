import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
import yaml
import sys
import copy

sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dt_dataloaders

# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion, scaler):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        if scaler is not None:
            out_batch = scaler.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)

                
                

@torch.no_grad()
def predict_with_similarity_euclidean(model, loader, scaler):
    model.eval()
    all_similarities = []

    
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        linkids =  x_batch[0, 0, :, 0]
        node_ids_list = [str(int(node_id)) for node_id in linkids]
        # 前向传播得到模型输出和 adaptive_embedding
        out_batch = model(x_batch)
        adaptive_embedding = model.adaptive_embedding  # torch.Size([12, 400, 40])

        # 合并时间步和embedding维度 -> (400, 480)
        node_embeddings = adaptive_embedding.view(adaptive_embedding.shape[1], -1)

        # 计算欧几里得距离矩阵 (400, 400)
        similarity_matrix = torch.cdist(node_embeddings, node_embeddings, p=2)
        
        all_similarities.append(similarity_matrix.cpu().numpy())

    # 计算所有batch的相似度矩阵的均值 (400, 400)
    mean_similarity_matrix = np.mean(all_similarities, axis=0)

    # 获取上三角矩阵的索引（避免重复计算相似度对）
    triu_indices = np.triu_indices_from(mean_similarity_matrix, k=1)
    flattened_distances = mean_similarity_matrix[triu_indices]

    # 找到最相关的20个节点对的索引（最小的距离）
    top_20_indices = np.argsort(flattened_distances)[:20]
    top_20_pairs = [(triu_indices[0][i], triu_indices[1][i]) for i in top_20_indices]
    with open("top_20_similar_pairs.txt", "w") as f:
        for idx, (node1, node2) in enumerate(top_20_pairs):
            f.write(f"Rank {idx + 1}: Node {node_ids_list[node1]} - Node {node_ids_list[node2]}, Distance: {flattened_distances[top_20_indices[idx]]:.4f}\n")
    return mean_similarity_matrix, top_20_pairs



@torch.no_grad()
def test_model(model, testset_loader, scaler, log=None):
    model.eval()
    

    start = time.time()
    y_true, y_pred = predict(model, testset_loader, scaler)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)
    return rmse_all, mae_all, mape_all


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #
    dt = '20240701'

    set_cpu_num(1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")


    with open(f"config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg["STAEFormer"]
    cfg = cfg["didi"]

    
    
    
    # --------------------------- set train, val, test data path -------------------------- #
    
    # 生成 dt_list
    dt = "20240701"
    dt_list = [(datetime.datetime.strptime(dt, '%Y%m%d') + datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range(90)]

    # 计算划分点
    total_len = len(dt_list)
    train_len = int(total_len * 0.6)
    val_len = int(total_len * 0.2)

    # 划分
    train_list = dt_list[:train_len]
    val_list = dt_list[train_len:train_len + val_len]
    test_list = dt_list[train_len + val_len:]
    
    # -------------------------------- load model -------------------------------- #
    from STAEFormer.model import STAEformer
    model = STAEformer(**cfg["model_args"])
    state_dict = torch.load(f"./saved_models/STAEFormer/STAEFormer-didi-2024-11-13-16-31-29.pt")
    model.load_state_dict(state_dict)
    
    testset_loader, scaler = get_dt_dataloaders(dt)
    model = model.to(DEVICE)
    mean_similarity_matrix, top_20_pairs = predict_with_similarity_euclidean(model, testset_loader, scaler)

    


