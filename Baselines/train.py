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

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, scaler, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    for batch in trainset_loader:
        x_batch, y_batch = batch 
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        if scaler is not None:
            out_batch = scaler.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
    model,
    model_name,
    train_list,
    val_list,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=10,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)
    # model = nn.DataParallel(model)

    wait = 0
    min_val_loss = np.inf
    
    train_loss_list = []
    val_loss_list = []
    # dt_list = [(datetime.strptime(dt, '%Y%m%d') + datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range(110)]
    for epoch in range(max_epochs):
        # train_total = 0
        val_total = 0
        train_total = 0
        for dt in train_list:
            print(f"Loading {dt} data...")
            if model_name == 'PatchSTG':
                trainset_loader, scaler = loadData(dt)
            else:
                trainset_loader, scaler = get_dt_dataloaders(dt, model=model_name)
            print(f"Training on {dt}...")
            train_cur = train_one_epoch(
                model, trainset_loader, optimizer, scheduler, criterion, clip_grad, scaler, log=log
            )
            print("Train Loss = %.5f" % train_cur)
            train_total += train_cur
            
        print("loading val data...")
        for dt in val_list:
            if model_name == 'PatchSTG':
                valset_loader, valset_loader_scaler = loadData(dt)
            else:
                valset_loader, valset_loader_scaler = get_dt_dataloaders(dt, model=model_name)
            val_cur = eval_model(model, valset_loader, criterion, valset_loader_scaler)
            print("Val Loss = %.5f" % val_cur)
            val_total += val_cur
        train_loss = train_total / len(train_list)
        train_loss_list.append(train_loss)
        val_loss = val_total / len(val_list)
        val_loss_list.append(val_loss)
        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

        model.load_state_dict(best_state_dict)
        train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader, scaler))
        val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader, scaler))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig(f"epoch_loss_{model_name}.png")

    if save:
        torch.save(best_state_dict, save)
    return model


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
    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"./logs/{model_name}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()
    
    # --------------------------- set model saving path -------------------------- #

    save_path = f"./saved_models/{model_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")
    
    
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
    elif model_name == "Attention":
        from Attention.model import AttnMLPModel
        model = AttnMLPModel(**cfg["model_args"])
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
        from SAGE.model import STID
        model = STID(**cfg["model_args"])
    elif model_name == "STID":
        from STID.model import STID
        model = STID(**cfg["model_args"])
    elif model_name == "PatchSTG":
        from PatchSTG.model import PatchSTG
        model = PatchSTG(**cfg["model_args"])
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
     
    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)


    valset_loader = None
    
    model = train(
        model,
        model_name,
        train_list,
        val_list,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 1),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log,
        save=save,
    )

    print_log(f"Saved Model: {save}", log=log)

    rmse_all, mae_all, mape_all = 0, 0, 0
    print("loading test data...")
    for dt in test_list:
        if model_name == "PatchSTG":
            testset_loader, scaler = loadData(dt)
        else:
            testset_loader, scaler = get_dt_dataloaders(dt ,model=model_name)
        print_log("--------- Test ---------", log=log)
        print_log(f"Test on {dt}", log=log)
        rmse_cur, mae_cur, mape_cur = test_model(model, testset_loader, scaler, log=log)
        rmse_all += rmse_cur
        mae_all += mae_cur
        mape_all += mape_cur
    rmse_all /= len(test_list)
    mae_all /= len(test_list)
    mape_all /= len(test_list)
    print_log(f"AVG RMSE: {rmse_all} MAE: {mae_all} MAPE: {mape_all}", log=log)
    log.close()
