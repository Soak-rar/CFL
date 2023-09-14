import copy

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
import HCFedAvg.DataGenerater

from ClusterMain import pca_dim_deduction


# 用于训练深度神经网络

def train(model, train_loader):

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i in range(2):
        for batch_index, (batch_data, batch_label) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            pred = model(batch_data)

            loss = F.nll_loss(pred, batch_label)

            loss.backward()

            optimizer.step()

    # for name, param in model1.named_parameters():
        # print(param.grad)
    model.to('cpu')
    return copy.deepcopy(model)


def test(model_, dataset_loader):
    model_.eval()
    test_loss = 0
    correct = 0
    count_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_.to(device)
    with torch.no_grad():
        for data, target in dataset_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            count_loss += 1
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_len = len(dataset_loader.dataset)
    test_loss /= test_len
    model_.to('cpu')
    model = None

    return test_loss, correct / test_len


def init_dataset_loader(dataset_name, batch_size):
    if dataset_name == 'mnist':
        test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))

        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'cifar10':
        test_dataset = datasets.CIFAR10(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset = datasets.CIFAR10(root='data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        pass


def avg(model_dict, local_model_dicts):
    total_len = len(local_model_dicts)

    for key in model_dict.keys():
        model_dict[key] *= 0
        for remote_model in local_model_dicts:
            model_dict[key] += (remote_model[key] / total_len)
    return model_dict


if __name__ == '__main__':
    clister_id = 0
    args = Args.Arguments()
    torch.random.manual_seed(3)
    init_model = Model.init_model(args.model_name)

    client_list = [i for i in range(0,90,5)]
    print(client_list)

    torch.save(init_model.state_dict(), 'HCFedAvg/test_model/init_model_dict.pth')

    AvgModel = copy.deepcopy(init_model)

    dataGen = HCFedAvg.DataGenerater.DatasetGen(args)

    test_data_loader = dataGen.get_cluster_test_DataLoader(clister_id)
    data_loader_list = {client: dataGen.get_client_DataLoader(client) for client in client_list}
    data_loader_95 = dataGen.get_client_DataLoader(95)
    data_loader_90 = dataGen.get_client_DataLoader(90)

    is_train = {client: False for client in client_list}

    signal_model_deep_list_95 = []
    signal_model_deep_list_90 = []
    avg_model_deep_list = []


    for i in range(100):
        train_clients = random.sample(client_list, 2)
        avg_list = []
        for client_id in train_clients:
            print(client_id)
            if is_train[client_id]:
                model_eval = train(AvgModel, data_loader_list[client_id])
            else:
                model_eval = train(copy.deepcopy(init_model), data_loader_list[client_id])
                is_train[client_id] = True

            avg_list.append(model_eval.state_dict())
        print('epoch: ', i)

        AvgModel.load_state_dict(avg(copy.deepcopy(init_model).state_dict(), avg_list))
        avg_model_deep_list.append(copy.deepcopy(AvgModel.state_dict()))
        if i % 5 == 0:
            sigal_model = copy.deepcopy(AvgModel)
            train(sigal_model, data_loader_95)
            signal_model_deep_list_95.append(copy.deepcopy(sigal_model.state_dict()))

            sigal_model = copy.deepcopy(AvgModel)
            train(sigal_model, data_loader_90)
            signal_model_deep_list_90.append(copy.deepcopy(sigal_model.state_dict()))

            loss, acc = test(sigal_model, test_data_loader)
            print('signal_model : ', 'acc: ', acc, ' loss', loss)


        loss, acc = test(AvgModel, test_data_loader)
        print('avg_model : ', 'acc: ', acc, ' loss', loss)


    torch.save(avg_model_deep_list, 'HCFedAvg/test_model/avg_model_deep_.pth')
    torch.save(signal_model_deep_list_95, 'HCFedAvg/test_model/signal_model_deep_95.pt')
    torch.save(signal_model_deep_list_90, 'HCFedAvg/test_model/signal_model_deep_90.pt')


