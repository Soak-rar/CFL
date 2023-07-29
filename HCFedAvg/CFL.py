
from DataGenerater import *

import copy

import Args

import torch.nn.functional as F
import numpy as np
import torch.optim as optim

import torch
import Model

from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import FileProcess


# 返回模型参数的更新值 global_model_dict 传入复制的模型字典
def train(global_model_dict, datasetLoader, worker_id, device, args: Args.Arguments):
    pre_model_param = global_model_dict
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

    local_model.to('cpu')
    update_model_param = local_model.state_dict()
    # 获取模型更新值
    for key in pre_model_param.keys():
        pre_model_param[key] = update_model_param[key] - pre_model_param[key]

    pre_model_param = trans_param_to_tensor(pre_model_param)

    return pre_model_param, update_model_param


def main(mArgs):
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    datasetGen = DatasetGen(mArgs)

    train_workers = [i for i in range(mArgs.worker_num)]

    # 每个客户端的 局部模型，初始化时为相同的模型

    global_model = Model.init_model(mArgs.model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() and mArgs.cuda else "cpu")

    clusters = {0:train_workers[:]}
    clusters_model = {0:copy.deepcopy(global_model.state_dict())}

    clients_in_cluster = np.zeros(mArgs.worker_num)

    clients_param = [None for _ in range(mArgs.worker_num)]
    clients_param_update = [None for _ in range(mArgs.worker_num)]

    EPS_1 = 0.75
    EPS_2 = 1.0
    warm_round = 20

    TotalLoss = []
    TotalAcc = []

    for global_round in range(mArgs.global_round):
        print("train round ", global_round)
        # 集群训练
        for cluster_id, cluster_model_dict in clusters_model.items():
            cluster_clients = clusters[cluster_id]
            for client_id in tqdm(cluster_clients, desc="Cluster_"+ str(cluster_id), unit="client", leave=True):
                train_loader = datasetGen.get_client_DataLoader(client_id)
                param_tensor_update, param_local = train(copy.deepcopy(cluster_model_dict), train_loader, client_id, device, mArgs)
                clients_param[client_id] = param_local
                clients_param_update[client_id] = param_tensor_update

        si_matrix = dt_matrix(clients_param_update)
        new_clusters = {}

        count = 0
        for cluster_id, clients_id in clusters.items():
            max_norm, mean_norm = compute_max_mean_update_norm([clients_param_update[client_id] for client_id in clients_id])
            print(max_norm)
            print(mean_norm)
            print('集群划分ing')
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(clients_id) > 2 and global_round > warm_round:
                print('达到条件')
                c1, c2 = divide_cluster_clients(si_matrix[clients_id][:, clients_id])

                c1, c2 = [clients_id[i] for i in c1], [clients_id[i] for i in c2]

                new_clusters[count] = c1
                new_clusters[count + 1] = c2
                count += 2

            else:
                print('未达到条件')
                new_clusters[count] = clients_id
                count += 1

        clusters = new_clusters
        print(clusters)
        # 更新集群模型
        for cluster_id, clients_id in clusters.items():
            for client_id in clients_id:
                clients_in_cluster[client_id] = cluster_id

            cluster_model_dict = copy.deepcopy(global_model.state_dict())
            for key in cluster_model_dict.keys():
                cluster_model_dict[key] *= 0
                for client_id in clients_id:
                    cluster_model_dict[key] += clients_param[client_id][key]
                cluster_model_dict[key] = cluster_model_dict[key]/len(clients_id)
            clusters_model[cluster_id] = cluster_model_dict

        round_acc = 0
        round_loss = 0
        for cluster_id, cluster_model_dict in clusters_model.items():
            new_cluster_id = find_cluster_id(clusters[cluster_id], mArgs.cluster_number) # 当前集群中客户端最多的
            ClusterDataLoader = datasetGen.get_cluster_test_DataLoader(new_cluster_id)
            cluster_model = Model.init_model(mArgs.model_name)
            cluster_model.load_state_dict(cluster_model_dict)
            loss, acc = test(cluster_model, ClusterDataLoader, device)
            round_acc += acc
            round_loss += loss

        TotalAcc.append(round_acc/len(clusters))
        TotalLoss.append(round_loss/len(clusters))
            # print(cluster_id, "  test")
        print("loss ", round_acc/len(clusters))
        print('acc ', round_loss/len(clusters))

        # 测试集群模型准确性
    save_dict = mArgs.save_dict()
    save_dict['algorithm_name'] = 'CFL'
    save_dict['acc'] = max(TotalAcc)
    save_dict['loss'] = min(TotalLoss)
    save_dict['traffic'] = 200 * 100
    save_dict['acc_list'] = TotalAcc
    save_dict['loss_list'] = TotalLoss

    FileProcess.add_row(save_dict)

def l2_dict(model_dict):
    l2_norm = torch.norm(model_dict, p=2)
    print(l2_norm)

def find_cluster_id(clients_list, cluster_num):
    clients_in_cluster = {cluster_id: 0 for cluster_id in range(cluster_num)}
    for client_id in clients_list:
        clients_in_cluster[client_id%cluster_num] += 1
    max_key = max(clients_in_cluster, key=clients_in_cluster.get)
    return max_key

def avg(model_dict, local_model_dicts):
    total_len = 0
    for model_inf in local_model_dicts:
        total_len += model_inf['data_len']
    for key in model_dict.keys():
        model_dict[key] *= 0
        for remote_model in local_model_dicts:
            model_dict[key] += (remote_model['model_dict'][key] * remote_model['data_len'] / total_len)
    return model_dict

def dt_matrix(param_list):
    a = b = torch.stack(param_list, dim=0)
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res

def divide_cluster_clients(S):
    clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

    c1 = np.argwhere(clustering.labels_ == 0).flatten()
    c2 = np.argwhere(clustering.labels_ == 1).flatten()
    return c1, c2

def compute_max_mean_update_norm(grad_list):
    a = torch.stack(grad_list, dim=0)
    return torch.max(a.norm(dim=1)).item(), torch.norm(torch.mean(a, dim=0)).item()

def trans_param_to_tensor(model_dict):
    parameters = [param.data.view(-1) for param in model_dict.values()]
    m_parameters = torch.cat(parameters)
    return m_parameters

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

if __name__ == '__main__':
    MyArgs = Args.Arguments()
    main(MyArgs)





