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
from lib.data_prepare import get_dt_dataloaders, loadData, readStaticData

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
def predict(model, loader, scaler):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        if scaler is not None:
            out_batch = scaler.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out)  # (samples, out_steps, num_nodes)
    y = np.vstack(y)

    return y, out



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
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="STAEFormer")
    parser.add_argument("-d", "--dataset", type=str, default="didi")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()

    seed = torch.randint(1000, (1,)) # set random seed here
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    dataset = args.dataset
    model_name = args.model_name
    print(f"Running {model_name} on {dataset} dataset...")
    with open(f"config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[model_name]
    cfg = cfg[dataset]
        
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
    if model_name == "STAEFormer":
        from STAEFormer.model import STAEformer
        model = STAEformer(**cfg["model_args"])
    elif model_name == "AR":
        from AutoRegression.model import ARModel
        model = ARModel(**cfg["model_args"])
    elif model_name == "MLP":
        from MLP.model import MLPModel
        model = MLPModel(**cfg["model_args"])
    elif model_name == "Linear":
        from MLP.model import LinearModel
        model = LinearModel(**cfg["model_args"])
    elif model_name == "DCST":
        from DCST.model import DCST
        model = DCST(**cfg["model_args"])
    elif model_name == "GWNet":
        from GWNet.model import gwnet
        model = gwnet(**cfg["model_args"])
    elif model_name == "AGCRN":
        from AGCRN.model import AGCRN
        model = AGCRN(**cfg["model_args"])
    elif model_name == "SAGE":
        from SAGE.model import PatchSTG
        model = PatchSTG(**cfg["model_args"])
        adj_path = "../data/dis_adj.npy"
        ori_parts_idx, reo_parts_idx, reo_all_idx = readStaticData(adj_path)
        model.set_index(ori_parts_idx, reo_parts_idx, reo_all_idx)
    elif model_name == "STID":
        from STID.model import STID
        model = STID(**cfg["model_args"])
    elif model_name == "PatchSTG":
        from PatchSTG.model import PatchSTG
        model = PatchSTG(**cfg["model_args"])
        adj_path = "../data/dis_adj.npy"
        ori_parts_idx, reo_parts_idx, reo_all_idx = readStaticData(adj_path)
        model.set_index(ori_parts_idx, reo_parts_idx, reo_all_idx)
    elif model_name == "PatchGCN":
        from PatchGCN.model import PatchGCN
        model = PatchGCN(**cfg["model_args"])
        adj_path = "../data/dis_adj.npy"
        ori_parts_idx, reo_parts_idx, reo_all_idx = readStaticData(adj_path)
        model.set_index(ori_parts_idx, reo_parts_idx, reo_all_idx)
    else:
        raise NotImplementedError

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = nn.HuberLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- train and test model --------------------------- #
     
    
    # 加载权重字典
    state_dict = torch.load('./saved_models/{}/PatchSTG-didi-2024-12-23-12-59-48.pt'.format(model_name))

    # 加载到模型
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    print("loading test data...")
    
    val_list += test_list
    # dt="20240910"
    for dt in val_list:
        if model_name == "PatchSTG":
            testset_loader, scaler, indices = loadData(dt, mode='test')
        else:
            testset_loader, scaler, indices = get_dt_dataloaders(dt, mode='test', model=model_name)
        y_true, y_pred = predict(model, testset_loader, scaler)
        rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
        out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            rmse_all,
            mae_all,
            mape_all,
        )
        print(y_true.shape)
        print(y_pred.shape)
        print(out_str)
        save_path = "./{}-/".format(model_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path += f"{dt}_predict.npy"
        y_pred = y_pred[torch.argsort(indices)]
        y_pred = y_pred.transpose(0, 2, 1, 3)
        np.save(save_path, y_pred)