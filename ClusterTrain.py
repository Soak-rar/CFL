import collections

import torch
import random
import math
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


def train(global_model_state_dict, dataset_dict, worker_id, local_loss_list, device, epoch, args):
    model = Model.init_model(args.model_name)
    model.load_state_dict(global_model_state_dict)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader, test_loader = init_local_dataloader(dataset_dict, args)
    model.train()
    LocalAvgGrad = torch.zeros(10, 20)
    LocalAvgParam = torch.zeros(10, 20)

    model.to(device)
    local_epoch_loss = 0
    for local_epoch in range(args.local_epochs):
        for batch_index, (batch_data, batch_label) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            pred = model(batch_data)

            loss = F.nll_loss(pred, batch_label)

            loss.backward()

            local_epoch_loss += loss.cpu().item() * (len(batch_data) / len(train_loader.dataset))

            # if local_epoch == args.local_epochs - 1:
            #     local_avg_grad = torch.zeros(20, requires_grad=True)
            #     for i in range(len(model.fc4.weight.grad)):
            #         local_avg_grad = local_avg_grad + model.fc4.weight.grad[i].cpu()
            #
            #     local_avg_grad /= len(model.fc4.weight.grad)
            #     LocalAvgGrad[] += (local_avg_grad * (len(batch_data) / len(train_loader.dataset)))

            for i in range(len(model.fc4.weight.grad)):
                LocalAvgGrad[i] += (model.fc4.weight.grad[i].cpu() * (len(batch_data) / len(train_loader.dataset)))


            optimizer.step()

        for i in range(len(model.fc4.weight)):
            LocalAvgParam[i] += model.fc4.weight[i].cpu()

    model.to('cpu')

    LocalAvgGrad /= args.local_epochs
    LocalAvgParam /= args.local_epochs
    local_loss_list[epoch] = local_epoch_loss
    data_len = sum(dataset_dict['data_len'].values())
    cost = 0
    cost += sum([param.nelement() for param in model.parameters()])

    loss, acc = test(model, test_loader, device)

    return {'model_dict': model.state_dict(),
            'data_len': data_len,
            'id': worker_id,
            'cost': cost * 4}, LocalAvgGrad, loss, acc, LocalAvgParam


def init_local_dataloader(dataset_dict, args):

    train_dataset = TensorDataset(dataset_dict['data'],
                                  dataset_dict['label'].to(torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(dataset_dict['data_test'],
                                 dataset_dict['label_test'].to(torch.long))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


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
    train_workers_dataset = Data.load_data(args)
    train_workers = [i for i in range(100)]
    global_model = Model.init_model(args.model_name)
    test_dataset_loader = init_test_dataset_loader(args.dataset_name, args.test_batch_size)

    cluster_models = {i: Model.init_model(args.model_name) for i in range(5)}

    local_client_loss = {i: collections.OrderedDict() for i in range(100)}

    cluster_clients = {i: train_workers[i:len(train_workers):5] for i in range(5)}

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # client_update_grad = {i: [] for i in range(len(train_workers))}
    # client_update_grad = {i: {} for i in range(len(train_workers))}

    client_model_param = {i: {} for i in range(len(train_workers))}

    # client_update_grad_with_ = {i: {} for i in range(len(train_workers))}

    cluster_loss = {i: {} for i in range(5)}
    cluster_acc = {i: {} for i in range(5)}
    for epoch in range(args.global_round):
        # print("Epoch :{}\t,".format(epoch + 1), end='')
        for cluster_id, cluster_model in cluster_models.items():
            cluster_local_models = []
            avg_loss = 0
            avg_acc = 0
            cluster_clients_train = random.sample(cluster_clients[cluster_id], 2)
            for worker_id in cluster_clients_train:
                print(epoch, "  worker : ", worker_id)
                train_eval, AvgGrad, loss, acc, AvgParam = train(cluster_model.state_dict().copy(), train_workers_dataset[cluster_id], worker_id,
                                                 local_client_loss[worker_id], device, epoch, args)
                # 根据loss 列表求平均梯度
                # loss 更新值， 大于0 记录
                # client_update_grad[worker_id][epoch] = AvgGrad
                client_model_param[worker_id][epoch] = AvgParam

                # if len(local_client_loss[worker_id]) > 1:
                #     differ_loss = {}
                #
                #     for i in range(len(local_client_loss[worker_id])-1):
                #         differ_l = local_client_loss[worker_id][list(local_client_loss[worker_id].keys())[i]] - local_client_loss[worker_id][list(local_client_loss[worker_id].keys())[i+1]]
                #         differ_loss[list(local_client_loss[worker_id].keys())[i+1]] = abs(differ_l)
                #     print(local_client_loss[worker_id])
                #     print(differ_loss)
                #     Loss_Sum = sum(differ_loss.values())
                #     LocalAvgGrad = torch.zeros(10, 20)
                #     for key_differ, value_differ in differ_loss.items():
                #         LocalAvgGrad = LocalAvgGrad + client_update_grad[worker_id][key_differ] * (value_differ / Loss_Sum)
                #     client_update_grad_with_[worker_id][epoch] = LocalAvgGrad
                #
                # else:
                #     client_update_grad_with_[worker_id][epoch] = AvgGrad

                cluster_local_models.append(train_eval)
                avg_loss += loss
                avg_acc += acc
            avg_loss /= len(cluster_clients_train)
            avg_acc /= len(cluster_clients_train)

            cluster_loss[cluster_id][epoch] = avg_loss
            cluster_acc[cluster_id][epoch] = avg_acc

            # print("Cluster:{}\t, Acc:{:.4f}, Loss:{:.5f} -- ".format(cluster_id+1, avg_acc, avg_loss), end='')

            avg_model_dict = avg(cluster_model.state_dict(), cluster_local_models)
            cluster_model.load_state_dict(avg_model_dict)
        print()
    SavePath = "DeepModelSimality" + '/' + ''
    # torch.save(client_update_grad,
    #            SavePath + '.pt')
    torch.save(cluster_loss,
               SavePath + '_ClusterLoss.pt')
    torch.save(cluster_acc,
               SavePath + '_ClusterAcc.pt')
    # torch.save(client_update_grad_with_,
    #            SavePath + '_weighting_grad.pt')
    torch.save(cluster_loss,
               SavePath + '_Cluster_Loss.pt')
    torch.save(client_model_param,
               SavePath + '_Param.pt')

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
