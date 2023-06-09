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

from ClusterMain import pca_dim_deduction


# 用于训练深度神经网络

def train(model, train_loader):

    optimizer = optim.SGD(model.parameters(), lr=0.2)
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    before_model_weight = model.state_dict()['fc4.weight'][:]

    model.to(device)
    sum = 0
    for batch_index, (batch_data, batch_label) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        sum += len(batch_data)
        pred = model(batch_data)

        loss = F.nll_loss(pred, batch_label)

        loss.backward()

        optimizer.step()
    print(len(train_loader.dataset))
    print(sum)
    # for name, param in model1.named_parameters():
        # print(param.grad)
    model.to('cpu')

    return model.fc4.weight.grad


def test(model, dataset_loader):
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


if __name__ == '__main__':
    train_loader1, test_loader1 = init_dataset_loader('mnist', 64)
    train_loader2, test_loader2 = init_dataset_loader('mnist', 64)
    model1 = Model.init_model('mnist')
    model2 = Model.init_model('mnist')
    loss_list1 = []
    acc_list1 = []
    model_update_param1 = []

    loss_list2 = []
    acc_list2 = []
    model_update_param2 = []
    for epoch in range(100):
        model_update1 = train(model1, train_loader1)
        model_update_param1.append(model_update1)
        loss1, acc1 = test(model1, test_loader1)

        loss_list1.append(loss1)
        acc_list1.append(acc1)

        model_update2 = train(model2, train_loader2)
        model_update_param2.append(model_update2)
        loss2, acc2 = test(model2, test_loader2)

        loss_list2.append(loss2)
        acc_list2.append(acc2)
        print('Global_Epoch: {}  ,  Loss 1: {:.5f},  Acc 1: {:.4f},  Loss 2: {:.5f},  Acc 2: {:.4f}\n'
              .format(epoch + 1, loss1, acc1, loss2, acc2, ))

        # pca_dim_deduction(np.array([model_update1[0]]))
        # print(KMeansPP.get_cos_dis_single_layer(pca_dim_deduction(np.array([model_update1[0].tolist()]), 3), pca_dim_deduction(np.array([model_update2[0].tolist()]), 3)))


    torch.save(model_update_param1,
               "DeepModelSimality" + '/' + 'param_grad_1_.pt')

    torch.save(model_update_param2,
               "DeepModelSimality" + '/' + 'param_grad_2_.pt')
