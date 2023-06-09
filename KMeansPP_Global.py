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


def get_cos_dis_single_layer(x, y):
    return torch.mean(torch.cosine_similarity(x, y, dim=-1))
    # return torch.cosine_similarity(x, y, dim=-1)


def get_cluster_center(clients_dict, center_clients):

    clients_to_center_dis = {}

    for client_id, client_data in clients_dict.items():
        if client_id in center_clients:
            continue
        dis_list = []
        for center_id in center_clients:
            dis = get_cos_dis_single_layer(client_data, clients_dict[center_id])
            dis_list.append(dis)

        clients_to_center_dis[client_id] = max(dis_list)

    for key, value in clients_to_center_dis.items():
        if value == min(clients_to_center_dis.values()):
            return key, value


def get_client_cluster(clusters_center, client_data):
    max_dis = -1
    cluster_id = 0
    for cluster_center_id, cluster_center in enumerate(clusters_center):
        dis = get_cos_dis_single_layer(cluster_center, client_data)
        if dis > max_dis:
            max_dis = dis
            cluster_id = cluster_center_id

    return cluster_id


def get_new_cluster_center(clusters, clients_dict):
    # 迭代每一个簇 ， 分别计算出每个 簇 的新的聚类中心
    clusters_center = []
    for cluster_id, cluster in enumerate(clusters):
        cluster_center = clients_dict[0].clone() * 0

        for client_in_cluster_id in cluster:
            cluster_center += clients_dict[client_in_cluster_id]

        cluster_center /= len(cluster)

        clusters_center.append(cluster_center)

    return clusters_center


def k_means_pp(clients_dict, args):

    # 初始化 k 个聚类中心
    center_clients = [random.randint(0, args.worker_num - 1)]

    for i in range(args.kMeansArgs.K-1):
        center_id, dis = get_cluster_center(clients_dict, center_clients)
        center_clients.append(center_id)

    clusters_center = [clients_dict[i] for i in center_clients]

    # 初始化 k 个簇集合（对应 k 个聚类中心）
    clusters = [[] for i in range(args.kMeansArgs.K)]
    for j in range(args.kMeansArgs.max_iter):
        clusters = [[] for i in range(args.kMeansArgs.K)]
        for client_id, client_data in clients_dict.items():
            # 计算每个点所属的 簇
            cluster_id = get_client_cluster(clusters_center, client_data)
            # 将 点 加入到对应的簇中
            # print(cluster_id)

            clusters[cluster_id].append(client_id)

        # 重新计算聚类中心
        clusters_center = get_new_cluster_center(clusters, clients_dict)

    return clusters


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
    return 0 if abs(x) < 0.001 else x


def clients_model_m_m(clients_dict):
    distence_m = [0 for i in range(len(clients_dict))]

    for worker_id_l in clients_dict.keys():
        distence_list = [0 for i in range(len(clients_dict))]
        for worker_id_r in clients_dict.keys():
            distence_list[worker_id_r] = round(get_cos_dis_single_layer(clients_dict[worker_id_l], clients_dict[worker_id_r]).item(),4)

        distence_m[worker_id_l] = distence_list
    return distence_m


def main(train_workers, args):
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

    distence_m = clients_model_m_m(clients_dict)
    head = ["distence_"+str(i) for i in range(args.worker_num)]
    index = ["client_"+str(i) for i in range(args.worker_num)]
    pd.DataFrame(columns=head, index=index, data=distence_m).to_csv(args.save_path + '/Global_clients_distence_distribution.csv')
    print(distence_m)
    #  k_means++
    clients_clusters = k_means_pp(clients_dict, args)
    data_frame = [{"簇"+str(i+1): 0 for i in range(args.kMeansArgs.K)} for j in range(10)]
    for i, clients_cluster in enumerate(clients_clusters):
        labels = {key: 0 for key in range(10)}
        for clients_id in clients_cluster:
            train_worker = train_workers[clients_id]
            for key, value in train_worker["data_len"].items():
                labels[key] += value
                data_frame[key]["簇"+str(i+1)] += value

        print("簇： {} \n 标签： {}".format(i, labels))
    pd.DataFrame(data_frame).to_csv(args.save_path + '/clusters_labels_distribution.csv')
    end_time = time.time()
    print("聚合时间 ： {}min {:.2f}second".format(int((end_time - start_time) / 60), (end_time - start_time) % 60))
    return clients_clusters, sum_cost


if __name__ == '__main__':
    args = Args.Arguments()
    train_workers = Data.load_data(args)
    for i in range(1):
        print('********* {} ***********'.format(i))
        main(train_workers, args)
