## 核心思路：指定 K 个集群，每次循环将 K 个集群的模型 全部发送给 参与训练的客户端， 客户端训练 K 个模型，
## 并根据 K 模型的损失 将 客户端指定为 损失最小的模型对应的集群中

import collections
import copy
from typing import *
import torch
import random
import math
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from DataGenerater import *

import Args
import KMeansPP
import datetime
import MMDLoss
import Model
import Data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import time
import os
import multiprocessing

from DataScientist import pca_deduce
from HCFedAvg import FileProcess
from HCFedAvg.CFL import trans_param_to_tensor


def train(global_model_dict: List, datasetLoader, worker_id, device, args: Args.Arguments):
    local_model = Model.init_model(args.model_name)
    local_model.load_state_dict(global_model_dict)

    if args.optim == 'Adam':
        optimizer = optim.Adam(local_model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(local_model.parameters(), lr=args.lr)

    local_model.to(device=device)
    local_model.train()
    loss_count = 0
    loss_sum = 0.0
    batch_num = 0

    for local_epoch in range(args.local_epochs):
        for batch_index, (batch_data, batch_label) in enumerate(datasetLoader):
            optimizer.zero_grad()

            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            pred = local_model(batch_data)
            loss = F.nll_loss(pred, batch_label)
            loss_sum += loss.item()
            loss_count += 1
            loss.backward()
            optimizer.step()

            if local_epoch == 0:
                batch_num += 1

    data_len = batch_num * args.batch_size
    local_model.to('cpu')
    update_model_param = local_model.state_dict()

    return update_model_param, data_len

def test(model, dataset_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    model = model.to(device)
    with torch.no_grad():
        for data, target in dataset_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_len = len(dataset_loader.dataset)
    test_loss /= test_len
    model.to('cpu')
    return test_loss, correct / test_len


def main(mArgs):
    dataGen = DatasetGen(mArgs)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    torch.manual_seed(5)

    global_model = Model.init_model(mArgs.model_name)

    train_workers = [i for i in range(args.worker_num)]

    TotalLoss = []
    TotalAcc = []

    for global_round in range(mArgs.global_round):
        worker_model_dicts = {}
        worker_data_len = {}

        clients_train = random.sample(train_workers, mArgs.worker_train)
        for worker_id in tqdm(clients_train, unit="client", leave=True):
            model_dict, data_len= train(global_model.state_dict(), dataGen.get_client_DataLoader(worker_id), worker_id, device, mArgs)
            worker_model_dicts[worker_id] = model_dict
            worker_data_len[worker_id] = data_len


        global_dict = global_model.state_dict()
        data_sum = sum(worker_data_len.values())
        for key in global_dict.keys():
            global_dict[key] *= 0
            for client_id in clients_train:
                global_dict[key] += worker_model_dicts[client_id][key] * (worker_data_len[client_id] * 1.0 / data_sum)

        global_model.load_state_dict(global_dict)
        test_dataloader = dataGen.get_fedavg_test_DataLoader()

        loss, acc = test(global_model, test_dataloader, device)

        TotalAcc.append(acc)
        TotalLoss.append(loss)
        # print(cluster_id, "  test")
        print()
        print(" epoch :  ", global_round)
        print("acc ", acc)
        print('loss ', loss)


    save_dict = mArgs.save_dict()
    save_dict['algorithm_name'] = 'FedAvg'
    save_dict['acc'] = max(TotalAcc)
    save_dict['loss'] = min(TotalLoss)
    save_dict['traffic'] = 200 * 10
    save_dict['acc_list'] = TotalAcc
    save_dict['loss_list'] = TotalLoss

    FileProcess.add_row(save_dict)

def find_cluster_id(clients_list, cluster_num):
    clients_in_cluster = {cluster_id: 0 for cluster_id in range(cluster_num)}
    for client_id in clients_list:
        clients_in_cluster[client_id%cluster_num] += 1
    max_key = max(clients_in_cluster, key=clients_in_cluster.get)
    return max_key


if __name__ == '__main__':
    args = Args.Arguments()
    main(args)
