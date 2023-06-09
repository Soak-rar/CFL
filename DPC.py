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
    distence_m = [0 for i in range(len(clients_dict))]
    dis_max = 0
    dis_min = 2
    for worker_id_l in clients_dict.keys():
        distence_list = [0 for i in range(len(clients_dict))]
        for worker_id_r in clients_dict.keys():
            distence_list[worker_id_r] = round(
                get_cos_dis_single_layer(clients_dict[worker_id_l], clients_dict[worker_id_r]).item(), 4)
            if worker_id_r != worker_id_l:
                if distence_list[worker_id_r] > dis_max:
                    dis_max = distence_list[worker_id_r]
                if distence_list[worker_id_r] < dis_min:
                    dis_min = distence_list[worker_id_r]
        distence_m[worker_id_l] = distence_list
    return distence_m, (dis_min, dis_max)


def get_density_matrix(distence_m, d_c):
    density_matrix = [0 for j in range(len(distence_m))]
    for i, dis_row in enumerate(distence_m):
        for j, point in enumerate(dis_row):
            if i != j:
                density_matrix[i] += get_point_density(point, d_c)
    print("density_matrix", density_matrix)
    return density_matrix


def get_delta(distence_m, density_matrix):
    delta_list = [-1 for i in range(len(density_matrix))]
    max_density = max(density_matrix)
    max_density_index = density_matrix.index(max_density)
    print(max_density_index)
    delta_list[max_density_index] = max(distence_m[max_density_index])
    delta_dict = {max_density_index: max_density_index}
    for i, i_density in enumerate(density_matrix):
        if i != max_density_index:
            dis_min = 2
            dis_min_idx = max_density_index
            for j, j_density in enumerate(density_matrix):
                if i != j and j_density > i_density and distence_m[i][j] < dis_min:
                    dis_min = distence_m[i][j]
                    dis_min_idx = j
            delta_list[i] = dis_min
            delta_dict[i] = dis_min_idx
    return delta_list, delta_dict


def get_point_density(dis_x_y, d_c):
    if (dis_x_y - d_c) > 0:
        return 0
    else:
        return 1


def get_point_counts(distence_m, d_c):
    in_count = 0
    out_count = 0
    for i in range(1, len(distence_m)):
        for j in range(i):
            if distence_m[i][j] < d_c:
                in_count += 1
            else : out_count += 1

    return in_count, out_count


def get_point_clusters(distence_m, d_c, centers_list):
    # 每个 点的 局部密度中 的其他点 的 id
    point_clusters = {i: [] for i in range(len(centers_list))}
    for i, (idx, _) in enumerate(centers_list):
        for j, dis in enumerate(distence_m[idx]):
            if idx != j and dis < d_c:
                point_clusters[i].append(j)
    return point_clusters


def get_cluster_center(density_matrix, delta_list, K):
    density_delta = {i: density_matrix[i] * delta_list[i] for i in range(len(density_matrix))}

    res = sorted(density_delta.items(), key=lambda d: d[1], reverse=True)
    return res[:K]


def main(train_workers, args, K):
    if os.path.exists(args.save_path + "global_distence_m.npy"):
        distence_m = np.load(args.save_path + "global_distence_m.npy").tolist()
        dis_min, dis_max = np.load(args.save_path + "global_dis_min_max.npy").tolist()
    else:
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

        distence_m, (dis_min, dis_max) = get_dis_matrix(clients_dict)
        np.save(args.save_path + "global_distence_m", distence_m)
        np.save(args.save_path + "global_dis_min_max", [dis_min, dis_max])
        print(dis_min, dis_max)
    print(dis_min, dis_max)
    # 计算dc
    dc = (dis_max + dis_min) / 2
    while True:
        n_neights, out_count = get_point_counts(distence_m, dc)
        rate = n_neights / (n_neights+out_count)
        if 0.01 <= rate <= 0.02:
            break
        if rate < 0.01:
            dis_min = dc
        else:
            dis_max = dc

        dc = (dis_max + dis_min) / 2

        if dis_max - dis_min < 0.0001:
            break
    print("dc: ", dc)

    density_matrix = get_density_matrix(distence_m, dc)

    delta_list, delta_dict = get_delta(distence_m, density_matrix)
    density_dict = {i: density_matrix[i] for i in range(len(density_matrix))}
    sorted_density_dict = sorted(density_dict.items(), key=lambda d: d[1], reverse=True)
    # 获取聚类中心
    centers_list = get_cluster_center(density_matrix, delta_list, K)
    print("centers_list: ", centers_list)
    data_frame1 = [{"簇" + str(i + 1): 0 for i in range(K)} for j in range(10)]
    for i, (center_id, _) in enumerate(centers_list):
        train_worker = train_workers[center_id]
        for key, value in train_worker["data_len"].items():
            data_frame1[key]["簇" + str(i + 1)] += value
    pd.DataFrame(data_frame1).to_csv(args.save_path + '/DPC_global_clusters_labels_centers_distribution_3.csv')

    # 获取聚类中心点的其余点的信息
    #     获取聚类每个点 局部半径 内的其余点
    # point_cluster = get_point_clusters(distence_m, dc, centers_list)
    # print(point_cluster.values())
    # data_frame1 = [{"簇" + str(i + 1): 0 for i in range(K)} for j in range(10)]
    # for i, center_list in point_cluster.items():
    #     for clients_id in center_list:
    #         train_worker = train_workers[clients_id]
    #         for key, value in train_worker["data_len"].items():
    #             data_frame1[key]["簇" + str(i + 1)] += value
    # pd.DataFrame(data_frame1).to_csv(args.save_path + '/DPC_clusters_labels_centers_distribution_3.csv')

    # 所有点聚类
    clusters = [-1 for i in range(args.worker_num)]
    for i, (i_id, dis) in enumerate(centers_list):
        clusters[i_id] = i

    for index, (den, _) in enumerate(sorted_density_dict):
        if clusters[den] == -1:
            clusters[den] = clusters[delta_dict[den]]

    print(clusters)
    clients_clusters_dict = {i: [] for i in range(K)}
    for i, cluster_id in enumerate(clusters):
        clients_clusters_dict[cluster_id].append(i)

    data_frame = [{"簇" + str(i + 1): 0 for i in range(K)} for j in range(10)]
    for i, clients_cluster in clients_clusters_dict.items():
        labels = {key: 0 for key in range(10)}
        for clients_id in clients_cluster:
            train_worker = train_workers[clients_id]
            for key, value in train_worker["data_len"].items():
                labels[key] += value
                data_frame[key]["簇" + str(i + 1)] += value

        print("簇： {} \n 标签： {}".format(i, labels))
    pd.DataFrame(data_frame).to_csv(args.save_path + '/DPC_global_clusters_labels_distribution_3.csv')

    plt.scatter(density_matrix, delta_list, marker="o")
    plt.xlabel("ρ")
    plt.ylabel("δ")
    plt.xticks(range(0, 15))
    plt.show()


if __name__ == '__main__':
    args = Args.Arguments()
    train_workers = Data.load_data(args)
    main(train_workers, args, 5)
