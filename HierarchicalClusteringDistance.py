import torch
import random
import Args
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
import pandas as pd
from matplotlib import pyplot as plt


def get_cos_dis_single_layer(x, y):
    return 1 - torch.mean(torch.cosine_similarity(x, y, dim=-1))
    # return torch.cosine_similarity(x, y, dim=-1)


def train(dataset_dict, worker_id, device, args):
    model = Model.init_model(args.model_name)

    local_dict = model.state_dict().copy()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = init_local_dataloader(dataset_dict, args)
    model.train()

    model = model.to(device)
    for local_epoch in range(args.kMeansArgs.local_epoch):
        for batch_index, (batch_data, batch_label) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            pred = model(batch_data)

            loss = F.nll_loss(pred, batch_label)

            loss.backward()

            optimizer.step()

    model.to('cpu')
    cost = 0
    for p in model.fc4.parameters():
        cost = p.nelement() * 4
        break

    return {'model_dict': model.state_dict()['fc4.weight'] - local_dict['fc4.weight'],
            'data_len': dataset_dict['data_len'],
            'id': worker_id,
            'cost': cost}


def init_local_dataloader(dataset_dict, args):
    train_dataset = TensorDataset(dataset_dict['data'],
                                  dataset_dict['label'].to(torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_loader


def filter_(x, *y):
    return 0 if abs(x) < 0.0001 else x


def get_dis_matrix(clients_dict):
    distence_m = {i: {j: 0 for j in range(len(clients_dict)) if j != i} for i in range(len(clients_dict))}
    for worker_id_l in distence_m.keys():
        for worker_id_r in distence_m[worker_id_l].keys():
            distence_m[worker_id_l][worker_id_r] = round(
                get_cos_dis_single_layer(clients_dict[worker_id_l], clients_dict[worker_id_r]).item(), 4)
    return distence_m


def main(train_workers, args, loadpath="HierarchicalClustering"):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    start_time = time.time()
    print("*********聚类*********")
    train_workers_id = range(args.worker_num)

    clients_dict = {}

    sum_cost = 0

    for worker_id in train_workers_id:
        train_eval = train(train_workers[worker_id], worker_id, device, args)
        clients_dict[worker_id] = train_eval["model_dict"].map_(train_eval["model_dict"], filter_)

        sum_cost += train_eval["cost"]

    dis_matrix = get_dis_matrix(clients_dict)
    print(dis_matrix)
    if not os.path.exists(loadpath):
        os.mkdir(loadpath)

    dir_path_id = 0
    while True:
        save_path = loadpath + "/dis_matrix_" + str(dir_path_id)+".npy"
        if not os.path.exists(save_path):
            np.save(save_path, dis_matrix)
            np.save(loadpath+"/clients_dict_" + str(dir_path_id) + ".npy", clients_dict)
            break
        else:
            dir_path_id += 1


if __name__ == '__main__':
    args = Args.Arguments()
    train_workers = Data.load_data(args)
    main(train_workers, args)
