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
from torch.multiprocessing import Manager
import torch.multiprocessing
import multiprocessing
import DataScientist
from DPC import get_dis_matrix, get_point_counts, get_density_matrix, get_delta
import DPC


def get_cos_dis_single_layer(x, y):
    rest = torch.cosine_similarity(torch.flatten(x), torch.flatten(y), dim=-1)
    return 1 - rest
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

        clients_to_center_dis[client_id] = min(dis_list)

    for key, value in clients_to_center_dis.items():
        if value == max(clients_to_center_dis.values()):
            return key, value


def get_client_cluster(clusters_center, client_data):
    max_dis = 2
    cluster_id = 0
    for cluster_center_id, cluster_center in enumerate(clusters_center):
        dis = get_cos_dis_single_layer(cluster_center, client_data)
        if dis < max_dis:
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

    for i in range(args.kMeansArgs.K - 1):
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

    return clusters, clusters_center


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


def train_global(dataset_dict, worker_id, device, args, global_model):
    model = Model.init_model(args.model_name)
    model.load_state_dict(global_model.state_dict().copy())
    local_dict = global_model.state_dict().copy()

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


def filter_(x, *y):
    return 0 if abs(x) < 0.0001 else x


def clients_model_m_m(clients_dict):
    distence_m = [0 for i in range(len(clients_dict))]

    for worker_id_l in clients_dict.keys():
        distence_list = [0 for i in range(len(clients_dict))]
        for worker_id_r in clients_dict.keys():
            distence_list[worker_id_r] = round(
                get_cos_dis_single_layer(clients_dict[worker_id_l], clients_dict[worker_id_r]).item(), 4)

        distence_m[worker_id_l] = distence_list
        distence_m[worker_id_l][worker_id_l] = 0
    return distence_m


def main(train_workers, args):
    pool = multiprocessing.Pool(1)
    queue = Manager().Queue()
    # args = Args.Arguments()
    # train_workers = Data.load_data(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    #
    start_time = time.time()
    print("*********聚类*********")
    train_workers_id = range(args.worker_num)

    clients_dict = {}

    sum_cost = 0
    print("---------初始模型训练---------")
    model = Model.init_model(args.model_name)
    for worker_id in train_workers_id:
        train_eval = train_global(train_workers[worker_id], worker_id, device, args, model)
        clients_dict[worker_id] = train_eval["model_dict"].map_(train_eval["model_dict"], filter_)

        sum_cost += train_eval["cost"]

    # style_list = ["b", "g", "r", "c", "m"]
    #
    # model_data = [DataScientist.pca_deduce(clients_dict[i].tolist(), 1) for i in range(len(clients_dict))]
    # model_data = DataScientist.pca_dim_deduction(model_data, 3)
    # print(model_data[0])
    # pool.apply_async(func=DataScientist.draw_3d_points, args=(model_data, style_list))
    # pool.close()
    # pool.join()
    # DataScientist.draw_3d_points(model_data, style_list)
    print("---------计算模型距离---------")
    distence_m = clients_model_m_m(clients_dict)
    head = ["distence_" + str(i) for i in range(args.worker_num)]
    index = ["client_" + str(i) for i in range(args.worker_num)]
    pd.DataFrame(columns=head, index=index, data=distence_m).to_csv(
        args.save_path + '/clusters_clients_distence_distribution_2.csv')
    #  k_means++
    print("---------集群划分---------")
    clients_clusters, cluster_centers = k_means_pp(clients_dict, args)
    data_frame = [{"簇" + str(i + 1): 0 for i in range(args.kMeansArgs.K)} for j in range(10)]
    for i, clients_cluster in enumerate(clients_clusters):
        labels = {key: 0 for key in range(10)}
        for clients_id in clients_cluster:
            train_worker = train_workers[clients_id]
            for key, value in train_worker["data_len"].items():
                labels[key] += value
                data_frame[key]["簇" + str(i + 1)] += value

        print("簇： {} \n 标签： {}".format(i, labels))
    pd.DataFrame(data_frame).to_csv(args.save_path + '/clusters_labels_distribution_2.csv')
    end_time = time.time()
    print("聚合时间 ： {}min {:.2f}second".format(int((end_time - start_time) / 60), (end_time - start_time) % 60))
    # DataScientist.draw_3d_points(model_data, style_list)
    # 集群划分完后 使用DPC筛选高密度集群
    # for i, clients_cluster in enumerate(clients_clusters):
    #     cluster_clients_dict = {}
    #     for client_id in clients_cluster:
    #         cluster_clients_dict[client_id] = clients_dict[client_id]
    #
    #     distence_m, (dis_min, dis_max) = get_dis_matrix(cluster_clients_dict)
    #     np.save(args.save_path + "global_distence_m", distence_m)
    #     np.save(args.save_path + "global_dis_min_max", [dis_min, dis_max])
    #     print(dis_min, dis_max)
    #     print(dis_min, dis_max)
    #     # 计算dc
    #     dc = (dis_max + dis_min) / 2
    #     while True:
    #         n_neights, out_count = get_point_counts(distence_m, dc)
    #         rate = n_neights / (n_neights + out_count)
    #         if 0.01 <= rate <= 0.02:
    #             break
    #         if rate < 0.01:
    #             dis_min = dc
    #         else:
    #             dis_max = dc
    #
    #         dc = (dis_max + dis_min) / 2
    #
    #         if dis_max - dis_min < 0.0001:
    #             break
    #     print("dc: ", dc)
    #
    #     density_matrix = get_density_matrix(distence_m, dc)
    #
    #     delta_list, delta_dict = get_delta(distence_m, density_matrix)
    #     density_dict = {i: density_matrix[i] for i in range(len(density_matrix))}
    #     sorted_density_dict = sorted(density_dict.items(), key=lambda d: d[1], reverse=True)
    #     # 获取聚类中心
    #     centers_list = DPC.get_cluster_center(density_matrix, delta_list, 1)
    #     print("centers_list: ", centers_list)

    return clients_clusters, cluster_centers, clients_dict


if __name__ == '__main__':
    args = Args.Arguments()
    train_workers = Data.load_data(args)
    main(train_workers, args)
