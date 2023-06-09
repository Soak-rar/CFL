import torch
import random

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
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


def train(global_model_state_dict, dataset_dict, worker_id, device, epoch, args):
    model = Model.init_model(args.model_name)
    model.load_state_dict(global_model_state_dict)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = init_local_dataloader(dataset_dict, args)
    model.train()

    model.to(device)
    for local_epoch in range(args.local_epochs):
        for batch_index, (batch_data, batch_label) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            pred = model(batch_data)

            loss = F.nll_loss(pred, batch_label)

            loss.backward()

            optimizer.step()

    model.to('cpu')


    data_len = sum(dataset_dict['data_len'].values())
    cost = 0
    cost += sum([param.nelement() for param in model.parameters()])

    return {'model_dict': model.state_dict(),
            'data_len': data_len,
            'id': worker_id,
            'cost': cost * 4}


def init_local_dataloader(dataset_dict, args):
    train_dataset = TensorDataset(dataset_dict['data'],
                                  dataset_dict['label'].to(torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_loader


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


def avg(model_dict, local_model_dicts):
    total_len = 0
    for model_inf in local_model_dicts:
        total_len += model_inf['data_len']
    for key in model_dict.keys():
        model_dict[key] *= 0
        for remote_model in local_model_dicts:
            model_dict[key] += (remote_model['model_dict'][key] * remote_model['data_len'] / total_len)
    return model_dict


def update_clusters_and_cluster_center(clients_clusters, cluster_centers, clients_dict_old, clients_dict_new):
    # 将集群中心 去掉要比较的 当前模型参数
    cluster_centers_filter = {}
    for worker_id, local_model_dict in clients_dict_new.items():
        for cluster_id, client_cluster in enumerate(clients_clusters):
            if worker_id in client_cluster:
                cluster_centers_filter[cluster_id] = cluster_centers[cluster_id] - clients_dict_old[worker_id] / len(client_cluster)
                client_cluster.remove(worker_id)
                break

    # 计算 每个 新局部模型到 新集群中心 cluster_centers_filter 的距离， 并重新划分集群
    for worker_id, local_model_dict in clients_dict_new.items():
        max_dis = 2
        belong_cluster_id = -1
        for cluster_id, cluster_center in cluster_centers_filter.items():
            cluster_dis = KMeansPP.get_cos_dis_single_layer(local_model_dict, cluster_center)
            if cluster_dis < max_dis:
                max_dis = cluster_dis
                belong_cluster_id = cluster_id
        print("客户端 {} 新所属集群: {} \n".format(worker_id, belong_cluster_id))
        clients_clusters[belong_cluster_id].append(worker_id)
        clients_dict_old[worker_id] = local_model_dict
        cluster_centers[belong_cluster_id] = ((cluster_centers_filter[belong_cluster_id] *
                                              (len(clients_clusters[belong_cluster_id]) - 1))
                                              + local_model_dict) / len(clients_clusters[belong_cluster_id])
    print("---------集群中的客户端数目--------")
    for cluster_id, client_cluster in enumerate(clients_clusters):
        print("集群  {} : {} \n".format(cluster_id, len(client_cluster)))
    print("-------------------------------")


def pca_dim_deduction(high_dim_data, max_dim):
    pca = PCA(n_components=max_dim, whiten=True)
    return pca.fit_transform(high_dim_data)


def draw_dynamic(clients_dict, clients_clusters):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot()
    plt.title("10-Class Data Distribution")

    markers = []
    colors = []

    while True:

        data_ = [[] for i in range(len(clients_dict))]

        if len(data_) > 0:
            keys = []
            values = []
            for key, value in clients_dict.items():
                keys.append(key)
                values.append(value[0].tolist())
            print(" values : ")

            new_value = pca_dim_deduction(np.array(values), 3)

            for i, id in enumerate(keys):
                data_[id] = new_value[i]
            print("clients_clusters : ")
            print(clients_clusters)
            point_style_list = ['lightcoral', 'darkkhaki', 'green', 'lightblue', 'mistyrose']
            for id, workers_id in enumerate(clients_clusters):
                for worker_id in workers_id:
                    ax.scatter(data_[worker_id][0], data_[worker_id][1], data_[worker_id][2], c=point_style_list[id], marker="^")
        plt.draw()
        plt.pause(1)
        ax.clear()


def filter_(x, *y):
    return 0 if abs(x) < 0.0001 else x


def main(args):
    train_workers = Data.load_data(args)
    global_model = Model.init_model(args.model_name)
    test_dataset_loader = init_test_dataset_loader(args.dataset_name, args.test_batch_size)

    local_models = {i: global_model.state_dict().copy() for i in range(len(train_workers))}
    clients_dict = multiprocessing.Manager().dict()
    clients_clusters = multiprocessing.Manager().list()

    draw_p = multiprocessing.Process(target=draw_dynamic, args=(clients_dict, clients_clusters))
    draw_p.start()
    # clients_clusters, sum_cost = KMeansPP.main(train_workers, args)

    PPclients_clusters, cluster_centers, PPclients_dict = KMeansPP.main(train_workers, args)

    for key, value in PPclients_dict.items():
        clients_dict[key] = value

    for value in PPclients_clusters:
        clients_clusters.append(value)

    # 集群划分后细化
    print("集群中心店点 -------------\n")
    print(cluster_centers)
    global_loss_list = []
    global_acc_list = []
    global_cost = []

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    print("---------联邦训练---------")
    start_time = time.time()
    for epoch in range(args.global_round):
        train_workers_id = []
        for clients_cluster in clients_clusters:
            train_workers_id.extend(random.sample(clients_cluster, args.cluster_worker_train))

        local_models = []

        for worker_id in train_workers_id:
            train_eval = train(global_model.state_dict().copy(), train_workers[worker_id], worker_id, device, epoch,
                               args)
            local_models.append(train_eval)
            # sum_cost += train_eval['cost']
        avg_model_dict = avg(global_model.state_dict(), local_models)
        global_model.load_state_dict(avg_model_dict)

        global_loss, global_acc = test(global_model, test_dataset_loader, device)
        global_loss_list.append(global_loss)
        global_acc_list.append(global_acc)
        # global_cost.append(sum_cost)

        # 当前训练全局模型聚合后，在这里 更新集群的划分
        ## 根据local_models构建 worker_id: model_dict 字典
        # local_id_models_dict = {worker['id']: (worker['model_dict']['fc4.weight'] - global_model.state_dict()['fc4.weight']).map_(worker['model_dict']['fc4.weight'] - global_model.state_dict()['fc4.weight'], filter_) for worker in
        #                         local_models}
        # update_clusters_and_cluster_center(clients_clusters, cluster_centers, clients_dict, local_id_models_dict)

        epoch_time = time.time()
        # 输出一次epoch的指标
        print('Global_Epoch: {}  ,  Loss: {:.5f},  Acc: {:.4f},  Epoch_Total_Time: {}min {:.2f}second\n'
              .format(epoch + 1, global_loss, global_acc * 100, int((epoch_time - start_time) / 60),
                      (epoch_time - start_time) % 60))

    global_test_eval = {'acc': global_acc_list, 'loss': global_loss_list}

    save(global_test_eval, global_cost, global_model.state_dict(), args)
    draw_p.terminate()


def init_test_dataset_loader(dataset_name, batch_size):
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        pass


def save(global_test_eval, global_cost, model_dict, args):
    args.to_string('FedPro')
    dir_path = args.save_path + '/' + 'Experiment'
    dir_path_id = 0
    while True:
        save_path = dir_path + str(dir_path_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            break
        else:
            dir_path_id += 1

    torch.save(global_cost,
               save_path + '/' + 'Global_Cost.pt')
    torch.save(global_test_eval,
               save_path + '/' + 'Global.pt')
    torch.save(model_dict,
               save_path + '/' + 'Model_Dict.pt')

    f = open(save_path + '/实验描述', 'w', encoding='UTF-8')
    f.write(args.Arg_string)
    f.write('\n' + str(datetime.datetime.now()))
    f.close()


if __name__ == '__main__':
    args = Args.Arguments()
    main(args)
