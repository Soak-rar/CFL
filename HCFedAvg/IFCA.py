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


def train(global_model_state_dict, data_, label_, worker_id, device, args):
    model = Model.init_model(args.model_name)
    model.load_state_dict(global_model_state_dict)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = init_local_dataloader(data_, label_, args)
    model.train()

    model.to(device)
    local_epoch_loss = 0

    for local_epoch in range(args.local_epochs):
        # for batch_index, (batch_data, batch_label) in enumerate(train_loader):

        optimizer.zero_grad()
        batch_data = data_.to(device)
        batch_label = label_.to(device)

        pred = model(batch_data)

        loss = F.nll_loss(pred, batch_label)

        local_epoch_loss += loss.cpu().item() * (len(batch_data) / len(train_loader.dataset))

        loss.backward()

        optimizer.step()

    model.to('cpu')

    local_epoch_loss /= args.local_epochs
    data_len = len(data_)
    cost = 0
    cost += sum([param.nelement() for param in model.parameters()])
    return {'model_dict': model.state_dict(),
            'data_len': data_len,
            'id': worker_id,
            'cost': cost * 4,
            'local_loss': local_epoch_loss}


def train_assign_cluster(cluster_models: Dict[int, nn.Module], data_, label_, worker_id, device, args):
    model = Model.init_model(args.model_name)
    min_local_epoch_loss = 99
    min_cluster_id = 0

    with torch.no_grad():
        for cluster_id, cluster_model in cluster_models.items():
            model.load_state_dict(cluster_model.state_dict())

            model.eval()
            model.to(device)
            # print(data_[0].dtype)
            # print(label_[0].dtype)
            # train_loader = init_local_dataloader(data_, label_, args)

            # batch_data, batch_label = next(iter(train_loader))

            batch_data = data_.to(device)
            batch_label = label_.to(device)

            pred = model(batch_data)

            loss = torch.nn.CrossEntropyLoss()(pred, batch_label)
            # print(loss)
            _correct = n_correct(pred, batch_label)

            # loss = F.nll_loss(pred, batch_label)

            local_epoch_loss = loss.cpu().item()

            if local_epoch_loss < min_local_epoch_loss:
                min_local_epoch_loss = local_epoch_loss
                min_cluster_id = cluster_id

    model.to('cpu')

    return min_cluster_id

def n_correct(y_logit, y):
    _, predicted = torch.max(y_logit.data, 1)
    correct = (predicted == y).sum().item()

    return correct

def init_local_dataloader(data_, label_, args):
    train_dataset = TensorDataset(data_,
                                  label_)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # test_dataset = TensorDataset(dataset_dict['data_test'],
    #                              dataset_dict['label_test'].to(torch.long))
    #
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_loader


def test(model, data_, label_, device):
    model.eval()
    test_loss = 0
    correct = 0
    model = model.to(device)
    with torch.no_grad():
        data = data_.to(device)
        target = label_.to(device)
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    model.to('cpu')
    return loss.cpu().item(), correct / len(data_)


def avg(model_dict, local_model_dicts):
    if len(local_model_dicts) == 0:
        return model_dict
    total_len = 0
    for model_inf in local_model_dicts:
        total_len += model_inf['data_len']
    for key in model_dict.keys():
        model_dict[key] *= 0
        for remote_model in local_model_dicts:
            model_dict[key] += (remote_model['model_dict'][key] * remote_model['data_len'] / total_len)
    return model_dict


def main(mArgs):
    DataSet = Data.generate_rotated_data(mArgs)
    train_workers = [i for i in range(mArgs.worker_num)]
    global_model = Model.init_model(mArgs.model_name)
    np.random.seed(3)
    torch.manual_seed(3)
    cluster_models: Dict[int, nn.Module] = {i: Model.init_model(mArgs.model_name) for i in range(args.cluster_number)}

    # for model in cluster_models.values():
    #     model.load_state_dict(global_model.state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() and mArgs.cuda else "cpu")

    for epoch in range(mArgs.global_round):
        # print("Epoch :{}\t,".format(epoch + 1), end='')

        # cluster_clients_train = random.sample(train_workers, args.worker_num)

        worker_in_cluster = [[]for i in range(args.cluster_number)]

        res = {i: 0 for i in range(args.cluster_number)}
        res_ = {i: 0 for i in range(args.cluster_number)}

        print('epoch : ', epoch)
        for worker_id in train_workers:
            # print('client: ', worker_id)
            worker_results = {}
            data_, label_ = load_data(worker_id, DataSet)
            in_cluster_id = train_assign_cluster(cluster_models, data_, label_, worker_id,
                                                             device, mArgs)
            print('current_cluster: ', DataSet['train']['cluster_assign'][worker_id], ', now_cluster', in_cluster_id)
            DataSet['train']['cluster_assign'][worker_id] = in_cluster_id
            train_eval = train(cluster_models[in_cluster_id].state_dict().copy(), data_, label_, worker_id,
                                                         device, mArgs)
            if DataSet['train']['cluster_assign'][worker_id] == in_cluster_id:
                res[in_cluster_id] += 1
            res_[in_cluster_id] += 1

            worker_in_cluster[in_cluster_id].append(train_eval)

        for cluster_id, local_models in enumerate(worker_in_cluster):
            cluster_models[cluster_id].load_state_dict(avg(cluster_models[cluster_id].state_dict(), local_models))
        print(res)
        print(res_)
        epoch_loss = []
        epoch_acc = []

        ## 集群准确性测试
        Loss = [[] for i in range(args.cluster_number)]
        Acc = [[] for i in range(args.cluster_number)]
        for worker_id in range(args.test_worker_num):
            data_, label_ = load_data(worker_id, DataSet, False)
            in_cluster_id = train_assign_cluster(cluster_models, data_, label_, worker_id,
                                                 device, mArgs)
            DataSet['test']['cluster_assign'][worker_id] = in_cluster_id
            loss, acc = test(cluster_models[in_cluster_id], data_, label_, device)
            Loss[in_cluster_id].append(loss)
            Acc[in_cluster_id].append(acc)
        print('Acc: ' , np.mean(Acc, axis=1).tolist())
        print('Loss: ' , np.mean(Loss, axis=1).tolist())
        # for cluster_id, model in cluster_models.items():
        #     test_dataset =
        #     test_dataloader = init_local_dataloader(test_dataset, args)
        #     test_model = Model.init_model(args.model_name)
        #     test_model.load_state_dict(model.state_dict())
        #     loss, acc = test(test_model, test_dataloader, device)
        #     print('cluster : ', cluster_id, ", loss: ", loss, ", acc: ", acc)
        #     epoch_loss.append(loss)
        #     epoch_acc.append(acc)


def load_data(m_i, dataset, train=True):
    # this part is very fast since its just rearranging models

    if train:
        dataset = dataset['train']
    else:
        dataset = dataset['test']

    indices = dataset['data_indices'][m_i]
    p_i = dataset['cluster_assign'][m_i]
    print(m_i, '  ', p_i)
    # print("indices, ", indices)
    X_batch = copy.deepcopy(dataset['X'][indices])
    y_batch = copy.deepcopy(dataset['y'][indices])

    # k : how many times rotate 90 degree
    # k =1 : 90 , k=2 180, k=3 270
    k = p_i

    X_batch2 = torch.rot90(X_batch, k=int(k), dims = (1,2))
    X_batch3 = X_batch2.reshape(-1, 28 * 28)

    # import ipdb; ipdb.set_trace()

    return X_batch3, y_batch


def get_worker_cluster_id(worker_results):
    min_loss = 99
    in_cluster = -1
    for cluster_id, train_eval in worker_results.items():
        if train_eval['local_loss'] <= min_loss:
            min_loss = train_eval['local_loss']
            in_cluster = cluster_id

    return in_cluster


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
