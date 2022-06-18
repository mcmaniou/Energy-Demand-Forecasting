from random import shuffle
import sys
import time
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.optim import optimizer
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import numpy as np
import model
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from functools import partial
import math
import scipy
import json
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import util
import model
plt.rcParams["font.size"] = 18

SEED = 42
util.fix_seed(SEED)
many = False

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE

    return ({'mape': mape, 'me': me, 'mae': mae,
             'mpe': mpe, 'rmse': rmse})

def make_dataloader( input_step, predict_step, use_clm, use_state, batch_size=1):
    dftest = pd.read_csv("test.csv", index_col=0)
    dftest = dftest  # [["total load actual"]]
    dftest.fillna(method='bfill', inplace=True)
    dftest.fillna(method='ffill', inplace=True)

    ################
    mu, sigma = 0, 1
    # creating a noise with the same dimension as the dataset (2,2)
    noise = np.random.normal(mu, sigma, len(dftest.index))
    dftest['Month'] = abs(6 - dftest['Month'])
    #dftest['temp'] = dftest['temp'] + noise

    dftest = dftest[dftest.columns[use_clm]]
    datatest = dftest.values

    dftrain = pd.read_csv("train.csv", index_col=0)
    dftrain = dftrain  # [["total load actual"]]
    dftrain.fillna(method='bfill', inplace=True)
    dftrain.fillna(method='ffill', inplace=True)

    ################
    mu, sigma = 0, 1
    # creating a noise with the same dimension as the dataset (2,2)
    noise = np.random.normal(mu, sigma, len(dftrain.index))
    dftrain['Month'] = abs(6 - dftrain['Month'])
    #dftrain['temp'] = dftrain['temp'] + noise

    dftrain = dftrain[dftrain.columns[use_clm]]
    datatrain = dftrain.values

    scaler = MinMaxScaler()
    scaler.fit(datatrain)

    dataset = scaler.transform(datatest)
    dataset = util.EnergyDataset(dataset, input_step, predict_step)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader

def main():
    parser = argparse.ArgumentParser(description='Test Voice Transformer Network')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_path', '-model', default="./checkout/model",type=str, help='root of model')
    parser.add_argument('--outpath', '-out', default="./checkout/test", type=str, help='root of output directory')
    parser.add_argument('--check_state', '-state', default="Alabama", type=str)
    # here train epochs minus 1
    parser.add_argument('--epoch_num', '-epn', default=80, type=int, help="学習済みモデルのエポック数")
    parser.add_argument('--dbg', '-dbg', action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    outpath = args.outpath
    check_state = args.check_state
    epoch_num = args.epoch_num

    config_path = os.path.join(model_path, "config/config.json")

    with open(config_path) as f:
        config = json.load(f)

    _, model_name = os.path.split(model_path)
    train_data_root = config["train_data_root"]
    test_data_root = config["test_data_root"]
    input_step = config["input_step"]
    predict_step = config["predict_step"]
    use_clm = config["use_clm"]
    d_model = config["d_model"]
    attn_type = config["attn_type"]
    N_enc = config["N_enc"]
    N_dec = config["N_dec"]
    h_enc = config["h_enc"]
    h_dec = config["h_dec"]
    ff_hidnum = config["ff_hidnum"]
    hid_pre = config["hid_pre"]
    hid_post = config["hid_post"]
    dropout_pre = config["dropout_pre"]
    dropout_post = config["dropout_post"]
    dropout_model = config["dropout_model"]
    use_state = config["use_state"]
    in_dim = len(use_clm)
    
    par_path = os.path.join(model_path, "model/{}_epoch.model".format(epoch_num))

    #######################################
    ############ Assert ###################
    #######################################

    if use_state != "all":
        if use_state[0] != check_state:
            assert False, "please set checkstate same as trained state"
    
    #########################################################################
    device = torch.device("cuda:{}".format(args.gpu)) if args.gpu >= 0 else torch.device("cpu")

    if use_state == "all":
        model_name = model_name+"_{}".format(check_state)
    util.set_directories(outpath, model_name, ["scatter", "forcast", "log"], args.dbg)
    log_path = os.path.join(outpath, model_name, "log")
    scatter_path = os.path.join(outpath, model_name, "scatter")
    forcast_path = os.path.join(outpath, model_name, "forcast")

    logger = util.Logger(log_path, "log", args_dict=config)

    test_set, test_loader = make_dataloader( input_step, predict_step, use_clm, [check_state], batch_size=1)
    all_data = test_set.data[:,0]

    net = model.Transformer(device, d_model, in_dim, attn_type, N_enc, N_dec, h_enc, h_dec, ff_hidnum, hid_pre, hid_post, dropout_pre, dropout_post, dropout_model)
    
    # load parameter
    if device == torch.device("cpu"):
        net.load_state_dict(torch.load(par_path, map_location=torch.device("cpu")))
    else:
        print("CUDA")
        net.load_state_dict(torch.load(par_path))
        net = net.to(device)

    net.eval()

    # predict
    # 出力するもの
    #  - scatter plot
    #  - 1時点先を予測し続けたplot
    pred_list = np.zeros((1, predict_step))  # (1, pred_len(1時点先, 2時点先))
    real=[]
    predtgt=[]
    for iter, (x, y, tgt, key) in enumerate(test_loader):
        x, y, tgt = x.to(device), y.to(device), tgt.to(device)
        tgt = tgt[:, :, 0]
        out = net.generate(x, tgt.shape[1], y[:,[0],:])  # (1(=batch), pred_len)
        if many:
            out=out[:,:,0]

        out = out.to('cpu').detach().numpy().copy()
        realv=tgt.to('cpu').detach().numpy().copy()
        real.append(realv[0,-1])
        predtgt.append(out[0,-1])

        pred_list = np.concatenate([pred_list,out ], axis=0)
        print(iter)
        if iter%100==0:
            print(forecast_accuracy(np.array(predtgt), np.array(real)))
    print(forecast_accuracy(np.array(predtgt),np.array(real)))
    pred_list = pred_list[1:]
    #D = []
    #for i in range(predict_step-1+23,predict_step+23):
    # tmp = np.concatenate(
    #         [all_data[:(input_step + 24)], pred_list[:, i], all_data[(all_data.shape[0] - (predict_step - (i + 1))):]])
    # D.append(tmp)

#     for step_num, tmp_data in enumerate(D):
#         n = step_num + 1
#
#         # scatter plot
#         corr = scipy.stats.pearsonr(tmp_data, all_data)
#         logger({"corr" : corr})
#
# #        np.save(os.path.join(scatter_path, "pred{}_epoch{}".format(n, epoch_num)), pred_list)
# #        np.save(os.path.join(scatter_path, "tgt"), tgt)
#
#         fig, ax = plt.subplots(figsize=(12,8))
#         ax.set_xlim(-1, 1)
#         ax.set_ylim(-1, 1)
#         ax.scatter(tmp_data,all_data)
#         ax.set_title("{} pred_{} epoch : {}".format(check_state, n, epoch_num))
#         ax.set_xlabel("predict")
#         ax.set_ylabel("true")
#         ax.text(-0.95, -0.5, "corr : {}".format(corr))
#         ax.axhline(0)
#         ax.axvline(0)
#         #plt.show()
# #        plt.savefig(os.path.join(scatter_path, "{}_pred{}_scatter_epoch{}.png".format(check_state,n,epoch_num)))
#        # plt.close()


    # forcast plot
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(real, label="true")

    #for ind, d in enumerate(D):
    #    ax.plot(d, label="forcast_{}".format(ind+1))
    ax.plot(predtgt, label="forcast_{}".format(24))
    plt.show()
    with open('prediction01467.npy', 'wb') as f:
        np.save(f, predtgt)
    with open('real.npy', 'wb') as f:
        np.save(f, real)
if __name__ == "__main__":
    main()
