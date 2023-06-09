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
    return torch.mean(torch.cosine_similarity(x, y, dim=-1))
    # return torch.cosine_similarity(x, y, dim=-1)


def train(dataset_dict, worker_id, device, args, model):
    # model = Model.init_model(args.model_name)

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

    return {'model_dict': model.state_dict()['fc4.weight']-local_dict['fc4.weight'],
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
    distence_m = [0 for i in range(len(clients_dict))]

    for worker_id_l in clients_dict.keys():
        distence_list = [0 for i in range(len(clients_dict))]
        for worker_id_r in clients_dict.keys():
            distence_list[worker_id_r] = round(
                get_cos_dis_single_layer(clients_dict[worker_id_l], clients_dict[worker_id_r]).item(), 4)

        distence_m[worker_id_l] = distence_list
    return distence_m


def get_density_matrix(distence_m, d_c):
    density_matrix = [0 for j in range(len(distence_m))]
    for i, dis_row in enumerate(distence_m):
        for j, point in enumerate(dis_row):
            if i != j:
                density_matrix[i] += get_point_density(point, d_c)

    return density_matrix


def get_delta(distence_m, density_matrix):
    delta_list = [-1 for i in range(len(density_matrix))]
    max_density = max(density_matrix)
    max_density_index = density_matrix.index(max_density)
    print(max_density_index)
    delta_list[max_density_index] = min(distence_m[max_density_index])
    for i, i_density in enumerate(density_matrix):

        if i != max_density_index:
            dis_max = 0
            for j, j_density in enumerate(density_matrix):
                if i != j and j_density > i_density and distence_m[i][j] > dis_max:
                    dis_max = distence_m[i][j]

            delta_list[i] = dis_max*-1
    return delta_list


def get_point_density(dis_x_y, d_c):
    return 1 if (dis_x_y - d_c) > 0 else 0


def main(train_workers, args, d_c):
    # args = Args.Arguments()
    # train_workers = Data.load_data(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    start_time = time.time()
    print("*********聚类*********")
    train_workers_id = range(args.worker_num)

    clients_dict = {}

    sum_cost = 0
    model = Model.init_model(args.model_name)

    for worker_id in train_workers_id:
        train_eval = train(train_workers[worker_id], worker_id, device, args, model)
        clients_dict[worker_id] = train_eval["model_dict"].map_(train_eval["model_dict"], filter_)

        sum_cost += train_eval["cost"]

    distence_m = get_dis_matrix(clients_dict)
    density_matrix = get_density_matrix(distence_m, d_c)
    delta_list = get_delta(distence_m, density_matrix)
    plt.scatter(density_matrix, delta_list, marker="o")
    plt.xticks(range(0, 100, 10))
    plt.show()
    print(get_delta)


if __name__ == '__main__':
    args = Args.Arguments()
    train_workers = Data.load_data(args)
    main(train_workers, args, 0.9)

